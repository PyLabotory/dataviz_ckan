# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 15:28:56 2025

@author: jburg
"""

# -*- coding: utf-8 -*-
#######PROYECTO FINAL - Taller de Programación#####################################################
#######DataViz con Python + APIs REST públicas (datos.gob.cl CKAN) + Streamlit#####################
#######Librerías exigidas: requests | json | pandas | matplotlib | streamlit#######################
###################################################################################################

import json                                #######Manejo de datos en formato JSON
import requests                            #######Peticiones HTTP GET a la API
import pandas as pd                        #######Análisis y manipulación de datos
import matplotlib.pyplot as plt            #######Gráficos con matplotlib
import streamlit as st                     #######Interfaz web interactiva con Streamlit
from datetime import datetime              #######Usado para mostrar fecha de consulta

#######CONFIGURACIÓN-DE-LA-APP########################################################################
st.set_page_config(page_title="DataViz Gobierno de Chile (CKAN)", layout="wide")

#######ENDPOINTS-BASE-CKAN############################################################################
CKAN_BASE = "https://datos.gob.cl/api/3/action"        #######URL base de la API CKAN

#######ESTADO-PERSISTENTE#############################################################################
if "results" not in st.session_state: st.session_state["results"] = []
if "dataset" not in st.session_state: st.session_state["dataset"] = None
if "recurso" not in st.session_state: st.session_state["recurso"] = None
if "resource_id" not in st.session_state: st.session_state["resource_id"] = None
if "df_conv" not in st.session_state: st.session_state["df_conv"] = None

#######FUNCIONES-AUXILIARES-API#######################################################################
def ckan_package_search(query: str, rows: int = 20, start: int = 0):
    #######Busca datasets por texto en CKAN (endpoint /package_search)###############################
    params = {"q": query, "rows": rows, "start": start}
    r = requests.get(f"{CKAN_BASE}/package_search", params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def ckan_datastore_search(resource_id: str, limit: int = 10000, offset: int = 0):
    #######Lee filas de un recurso con DataStore activo (endpoint /datastore_search)#################
    params = {"resource_id": resource_id, "limit": limit, "offset": offset}
    r = requests.get(f"{CKAN_BASE}/datastore_search", params=params, timeout=30)
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=False)
def fetch_full_resource(resource_id: str, max_rows: int = 100000):
    #######Descarga completa de un recurso con paginación############################################
    dfs = []
    limit = 32000
    total = None
    offset = 0
    while True:
        data = ckan_datastore_search(resource_id, limit=limit, offset=offset)
        if not data.get("success"):
            raise RuntimeError("API devolvió success=False en datastore_search")
        res = data["result"]
        records = res.get("records", [])
        if not records:
            break
        dfs.append(pd.DataFrame.from_records(records))
        total = res.get("total", None)
        offset += len(records)
        if total is not None and offset >= min(total, max_rows):
            break
        if offset >= max_rows:
            break
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

#######TÍTULO-Y-DESCRIPCIÓN##########################################################################
st.title("👑 Proyecto Final – DataViz con APIs del Gobierno de Chile")
st.caption("Aplicación construida en Python con Streamlit para Taller de Programación")

with st.expander("Descripción de la aplicación"):
    st.markdown("""
- Permite buscar datasets en datos.gob.cl.
- Selecciona recursos con DataStore activo (consultables por API).
- Descarga datos con peticiones GET y paginación.
- Analiza datos con pandas y los visualiza con matplotlib.
- Exporta a CSV y muestra metadatos del recurso.
""")

#######PASO-1:BUSCAR-DATASETS#########################################################################
st.subheader("1) Buscar datasets")
colq, colbtn = st.columns([3,1])
with colq:
    q = st.text_input("Palabra clave (ej: educación, transporte, salud)", value="educación", key="q_key")
with colbtn:
    buscar = st.button("Buscar", type="primary", key="buscar_btn")

results = st.session_state["results"]
if buscar and q.strip():
    try:
        pack = ckan_package_search(q.strip(), rows=20, start=0)
        if pack.get("success"):
            st.session_state["results"] = pack["result"]["results"]
            st.session_state["dataset"] = None
            st.session_state["recurso"] = None
            st.session_state["resource_id"] = None
            st.session_state["df_conv"] = None
            results = st.session_state["results"]
        else:
            st.error("La API devolvió success=False en package_search.")
    except Exception as e:
        st.error(f"Error al buscar datasets: {e}")

if results:
    st.write(f"Se encontraron **{len(results)}** datasets (mostrando hasta 20).")
    opciones = []
    idx_map = {}
    for ds in results:
        title = ds.get("title", "(sin título)")
        org = (ds.get("organization") or {}).get("title", "Desconocida")
        label = f"{title} — Org: {org}"
        opciones.append(label)
        idx_map[label] = ds

    ds_sel = st.selectbox(
        "Seleccione un dataset",
        opciones,
        index=0 if st.session_state["dataset"] is None else opciones.index(
            f"{st.session_state['dataset'].get('title','(sin título)')} — Org: {(st.session_state['dataset'].get('organization') or {}).get('title','Desconocida')}"
        ) if f"{st.session_state['dataset'].get('title','(sin título)')} — Org: {(st.session_state['dataset'].get('organization') or {}).get('title','Desconocida')}" in opciones else 0,
        key="dataset_select"
    )
    dataset = idx_map[ds_sel]
    st.session_state["dataset"] = dataset

    #######PASO-2:LISTAR-RECURSOS-CON-DATASTORE########################################################
    st.subheader("2) Elegir recurso consultable (DataStore activo)")
    recs = dataset.get("resources", []) or []
    recs_ds = [r for r in recs if r.get("datastore_active")]
    if not recs_ds:
        st.warning("Este dataset no tiene recursos con DataStore activo. Seleccione otro dataset.")
        st.stop()

    rec_labels = [f"{(r.get('name') or '(sin nombre)')} — resource_id={r.get('id')}" for r in recs_ds]
    rec_map = {lbl: r for lbl, r in zip(rec_labels, recs_ds)}

    if st.session_state["resource_id"] in [r.get("id") for r in recs_ds]:
        default_idx = [r.get("id") for r in recs_ds].index(st.session_state["resource_id"])
    else:
        default_idx = 0

    rec_sel = st.selectbox("Recurso consultable", rec_labels, index=default_idx, key="recurso_select")
    recurso = rec_map[rec_sel]
    resource_id = recurso.get("id")
    st.session_state["recurso"] = recurso
    st.session_state["resource_id"] = resource_id

    #######PASO-3:DESCARGAR-DATOS######################################################################
    st.subheader("3) Descargar datos")
    max_rows = st.slider("Máximo de filas a traer", min_value=1000, max_value=100000, value=50000, step=1000, key="slider_rows")
    cargar = st.button("Cargar datos", type="primary", key="cargar_btn")

    if cargar:
        try:
            df = fetch_full_resource(resource_id, max_rows=max_rows)
            if df.empty:
                st.warning("Recurso sin registros o no retornó datos.")
            else:
                df_conv = df.copy()
                for col in df_conv.columns:
                    low = str(col).lower()
                    if any(k in low for k in ["fecha", "date", "datetime", "periodo"]):
                        try:
                            df_conv[col] = pd.to_datetime(df_conv[col], errors="ignore")
                        except Exception:
                            pass
                    if df_conv[col].dtype == object:
                        try:
                            df_conv[col] = pd.to_numeric(
                                pd.Series(df_conv[col], dtype="object").astype(str).str.replace(",", ".", regex=False),
                                errors="ignore"
                            )
                        except Exception:
                            pass
                st.session_state["df_conv"] = df_conv
                st.success(f"Datos cargados: {len(df_conv):,} filas × {df_conv.shape[1]} columnas")
        except Exception as e:
            st.error(f"Error al cargar/analizar datos: {e}")

#######RENDERS-CUANDO-HAY-DATOS#######################################################################
df_conv = st.session_state["df_conv"]
recurso = st.session_state["recurso"]
dataset = st.session_state["dataset"]

if isinstance(df_conv, pd.DataFrame) and not df_conv.empty and recurso and dataset:
    c1, c2, c3 = st.columns(3)
    c1.metric("Filas", f"{len(df_conv):,}")
    c2.metric("Columnas", f"{df_conv.shape[1]}")
    c3.metric("Consulta", datetime.now().strftime("%Y-%m-%d %H:%M"))

    st.dataframe(df_conv.head(50), use_container_width=True)

    csv = df_conv.to_csv(index=False).encode("utf-8")
    st.download_button("Descargar CSV", data=csv, file_name=f"{st.session_state['resource_id']}.csv", mime="text/csv")

    #######PASO-4:FILTROS####################################################################
    st.subheader("4) Filtros de datos")
    cols = list(df_conv.columns)
    col_filtro = st.multiselect("Columnas a mostrar", cols, default=cols[:min(8, len(cols))])

    col_where = st.selectbox("Filtrar por columna (opcional)", ["(sin filtro)"] + cols)
    df_view = df_conv
    if col_where != "(sin filtro)":
        val_where = st.text_input("Valor a buscar")
        if val_where:
            serie = df_conv[col_where].astype(str)
            mask = serie.str.contains(val_where, case=False, na=False) | (serie == val_where)
            df_view = df_conv[mask]

    if col_filtro:
        st.dataframe(df_view[col_filtro].head(100), use_container_width=True)

    #######PASO-5:AGREGACIÓN#################################################################
    st.subheader("5) Agregación de datos")
    posibles_dim = [c for c in df_view.columns if (df_view[c].dtype == object or pd.api.types.is_datetime64_any_dtype(df_view[c]))]
    posibles_num = [c for c in df_view.columns if pd.api.types.is_numeric_dtype(df_view[c])]

    dim = st.selectbox("Columna de agrupación", ["(ninguna)"] + posibles_dim)
    medida = st.selectbox("Columna numérica", ["(ninguna)"] + posibles_num)
    aggfunc = st.selectbox("Función de agregación", ["suma", "promedio", "conteo"])

    agg_df = None
    if dim != "(ninguna)":
        if medida != "(ninguna)":
            if aggfunc == "suma":
                agg_df = df_view.groupby(dim, dropna=False)[medida].sum(numeric_only=True).reset_index()
            elif aggfunc == "promedio":
                agg_df = df_view.groupby(dim, dropna=False)[medida].mean(numeric_only=True).reset_index()
            else:
                agg_df = df_view.groupby(dim, dropna=False)[medida].count().reset_index(name="conteo").rename(columns={medida: "conteo"})
        else:
            agg_df = df_view.groupby(dim, dropna=False).size().reset_index(name="conteo")

    if agg_df is not None and not agg_df.empty:
        st.write("Resultado de la agregación (primeras 50 filas):")
        st.dataframe(agg_df.head(50), use_container_width=True)

        #######PASO-6:VISUALIZACIÓN#############################################################
        st.subheader("6) Visualización de resultados")
        chart_type = st.selectbox("Tipo de gráfico", ["Líneas", "Barras"])

        xcol = dim
        if medida != "(ninguna)" and aggfunc != "conteo":
            ycol = medida
        else:
            ycol = "conteo"

        if pd.api.types.is_datetime64_any_dtype(agg_df[xcol]):
            agg_df = agg_df.sort_values(xcol)

        fig, ax = plt.subplots()
        if chart_type == "Líneas":
            ax.plot(agg_df[xcol], agg_df[ycol])
        else:
            ax.bar(agg_df[xcol].astype(str), agg_df[ycol])
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.set_title(f"{chart_type} — {ycol} por {xcol}")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

    #######PASO-7:METADATOS##################################################################
    st.subheader("7) Metadatos del recurso")
    meta_cols = ["id", "name", "format", "mimetype", "size", "url"]
    meta_view = {k: recurso.get(k) for k in meta_cols if k in recurso}
    extra = {
        "dataset_title": dataset.get("title"),
        "organization": (dataset.get("organization") or {}).get("title"),
        "api_endpoints": {
            "package_search": f"{CKAN_BASE}/package_search",
            "datastore_search": f"{CKAN_BASE}/datastore_search?resource_id={st.session_state['resource_id']}"
        }
    }
    st.json({"resource_meta": meta_view, "dataset_info": extra})

    st.info("Se recomienda citar: título del dataset, organización, resource_id, URL y fecha de consulta.")

#######FOOTER#########################################################################################
st.markdown("---")
st.markdown("© 2025 · Proyecto académico – Taller de Programación · Universidad San Sebastián")
