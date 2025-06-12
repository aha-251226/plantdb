import os
import json
import base64
import logging
from io import BytesIO
import pdfplumber
import duckdb
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from dotenv import load_dotenv
load_dotenv()

client = None  # OpenAI 클라이언트는 외부에서 주입되거나 설정됨
DB_DUCKDB = "plant_data.duckdb"
EXPORT_DIR = "exported_csv"
os.makedirs(EXPORT_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO)

FIELD_MAPPING_RULES = {
    "document number": "doc_no",
    "doc. no.": "doc_no",
    "doc no": "doc_no",
    "job no": "job_no",
    "job number": "job_no",
    "project": "project",
    "item no": "item_no",
    "item number": "item_no",
}

DB_TABLES = {
    "project_meta": {
        "project_id": "INTEGER",
        "job_no": "VARCHAR", "project": "VARCHAR", "doc_no": "VARCHAR", "item_no": "VARCHAR",
        "client": "VARCHAR", "service": "VARCHAR", "page": "VARCHAR", "rev": "VARCHAR", "location": "VARCHAR"
    },
    "process_requirement": {
        "id": "INTEGER",
        "project_id": "INTEGER",
        "field_name": "VARCHAR", "am_feed": "VARCHAR", "ah_feed": "VARCHAR", "note": "VARCHAR"
    },
    "others": {
        "id": "INTEGER",
        "project_id": "INTEGER",
        "casing_class": "VARCHAR", "impeller_class": "VARCHAR", "shaft_class": "VARCHAR", "corrosion_allowance_mm": "FLOAT",
        "start_method_manual": "BOOLEAN", "start_method_automatic": "BOOLEAN",
        "pump_liquid_flammable": "BOOLEAN", "pump_liquid_solidification": "BOOLEAN", "pump_liquid_toxic": "BOOLEAN",
        "pump_liquid_h2s": "BOOLEAN", "pump_liquid_chloride": "BOOLEAN", "insulation": "BOOLEAN",
        "steam_tracing": "BOOLEAN", "steam_jacket": "BOOLEAN",
        "location_indoor": "BOOLEAN", "location_outdoor": "BOOLEAN", "location_under_roof": "BOOLEAN",
        "leakage_permissibility": "VARCHAR", "estimated_shutoff_pressure_kgcm2g": "FLOAT",
        "driver_steam_inlet": "BOOLEAN", "driver_steam_exhaust": "BOOLEAN"
    },
    "notes": {
        "note_id": "INTEGER",
        "project_id": "INTEGER",
        "note_text": "VARCHAR"
    },
    "revision_history": {
        "id": "INTEGER",
        "project_id": "INTEGER",
        "revision": "VARCHAR", "date": "VARCHAR", "by_checked": "VARCHAR"
    },
    "unknown": {
        "id": "INTEGER",
        "raw_json": "VARCHAR"
    }
}

def init_database():
    con = duckdb.connect(DB_DUCKDB)
    for t in ["revision_history", "notes", "others", "process_requirement", "project_meta", "unknown"]:
        con.execute(f"DROP TABLE IF EXISTS {t}")
    con.execute("""
        CREATE TABLE project_meta (
            project_id INTEGER,
            job_no VARCHAR, project VARCHAR, doc_no VARCHAR, item_no VARCHAR,
            client VARCHAR, service VARCHAR, page VARCHAR, rev VARCHAR, location VARCHAR
        )
    """)
    con.execute("""
        CREATE TABLE process_requirement (
            id INTEGER,
            project_id INTEGER,
            field_name VARCHAR, am_feed VARCHAR, ah_feed VARCHAR, note VARCHAR
        )
    """)
    con.execute("""
        CREATE TABLE others (
            id INTEGER,
            project_id INTEGER,
            casing_class VARCHAR, impeller_class VARCHAR, shaft_class VARCHAR, corrosion_allowance_mm FLOAT,
            start_method_manual BOOLEAN, start_method_automatic BOOLEAN,
            pump_liquid_flammable BOOLEAN, pump_liquid_solidification BOOLEAN, pump_liquid_toxic BOOLEAN,
            pump_liquid_h2s BOOLEAN, pump_liquid_chloride BOOLEAN, insulation BOOLEAN,
            steam_tracing BOOLEAN, steam_jacket BOOLEAN,
            location_indoor BOOLEAN, location_outdoor BOOLEAN, location_under_roof BOOLEAN,
            leakage_permissibility VARCHAR, estimated_shutoff_pressure_kgcm2g FLOAT,
            driver_steam_inlet BOOLEAN, driver_steam_exhaust BOOLEAN
        )
    """)
    con.execute("""
        CREATE TABLE notes (
            note_id INTEGER,
            project_id INTEGER,
            note_text VARCHAR
        )
    """)
    con.execute("""
        CREATE TABLE revision_history (
            id INTEGER,
            project_id INTEGER,
            revision VARCHAR, date VARCHAR, by_checked VARCHAR
        )
    """)
    con.execute("""
        CREATE TABLE unknown (
            id INTEGER,
            raw_json VARCHAR
        )
    """)
    con.close()

def extract_pdf_tables_from_bytes(pdf_bytes):
    tables = []
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            try:
                extracted = page.extract_tables()
                if not extracted:
                    logging.warning(f"No tables on page {i+1}")
                    continue
                for t in extracted:
                    if t and any(any(cell for cell in row) for row in t):
                        tables.append(t)
            except Exception as e:
                logging.warning(f"Error reading page {i+1}: {e}")
    if not tables:
        logging.error("⚠️ No tables found in any pages.")
    return tables

def classify_table_by_header(table):
    header_row = table[0]
    header_join = " ".join([str(h).lower() for h in header_row if h])
    if any(k in header_join for k in ["project", "job no", "doc. no"]):
        return "project_meta"
    elif any(k in header_join for k in ["am_feed", "ah_feed", "note", "field_name"]):
        return "process_requirement"
    elif any(k in header_join for k in ["casing", "impeller", "shaft", "corrosion"]):
        return "others"
    elif any(k in header_join for k in ["note", "remarks", "comments"]):
        return "notes"
    elif any(k in header_join for k in ["revision", "date", "checked"]):
        return "revision_history"
    else:
        return "unknown"

def apply_field_mapping_rules(field_names):
    mapped = []
    for f in field_names:
        f_low = f.lower().replace(" ", "").replace(".", "")
        found = None
        for k, v in FIELD_MAPPING_RULES.items():
            if k.replace(" ", "").replace(".", "") in f_low:
                found = v
                break
        mapped.append(found if found else f)
    return mapped

def convert_value_to_type(value, dtype):
    if value is None:
        return None
    val = str(value).strip()
    if dtype == "BOOLEAN":
        return val.lower() in ("true", "1", "yes", "y", "on", "checked", "t")
    elif dtype == "INTEGER":
        try:
            return int(val)
        except:
            return None
    elif dtype == "FLOAT":
        try:
            return float(val)
        except:
            return None
    else:
        return val

def parse_table_to_dict(table, field_names):
    rows = []
    for row in table[1:]:
        row_dict = {}
        for i, f in enumerate(field_names):
            row_dict[f] = row[i] if i < len(row) else None
        rows.append(row_dict)
    return rows

def gpt_table_field_mapping(table, docname, table_type_hint):
    prompt = f"""You are a highly reliable table header interpreter.

You are given a table extracted from a plant equipment specification PDF.
Your job is to analyze the **first row only** as a header and generate clean, concise, lowercase English field names for each column.

Instructions:
- Do NOT consider data rows (rows 2~5) for field name generation.
- Only base your answer on row 1 (the header row).
- Output a list of JSON-formatted field names.
- Strip units, symbols, or comments from the field name.
- Convert to lowercase, replace spaces with underscores.

Input (header row only):
{json.dumps(table[0])}

Output:
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=128,
            temperature=0,
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        logging.warning(f"GPT field mapping failed. Using first row as header. Reason: {e}")
        return [str(h).lower().replace(" ", "_") if h else f"col_{i}" for i, h in enumerate(table[0])]

def insert_table_data(table_name, project_id, rows, field_names):
    con = duckdb.connect(DB_DUCKDB)
    con.execute("BEGIN TRANSACTION")
    try:
        for row in rows:
            db_columns = list(DB_TABLES.get(table_name, {}).keys())
            if not db_columns:
                con.execute("INSERT INTO unknown (raw_json) VALUES (?)", (json.dumps(row),))
                continue

            mapped_row = {}
            for f in field_names:
                f_low = f.lower().replace(" ", "").replace(".", "")
                mapped_field = None
                for k, v in FIELD_MAPPING_RULES.items():
                    if k.replace(" ", "").replace(".", "") in f_low:
                        mapped_field = v
                        break
                if mapped_field is None and f in db_columns:
                    mapped_field = f
                if mapped_field:
                    mapped_row[mapped_field] = row.get(f, None)

            db_row = {}
            for col in db_columns:
                dtype = DB_TABLES[table_name].get(col, "VARCHAR").upper()
                val = mapped_row.get(col, None)
                db_row[col] = convert_value_to_type(val, dtype)

            if table_name == "project_meta":
                res = con.execute("""
                    SELECT project_id FROM project_meta WHERE job_no = ? AND doc_no = ? AND item_no = ?
                """, (db_row.get("job_no"), db_row.get("doc_no"), db_row.get("item_no"))).fetchone()

                if res:
                    con.execute("""
                        UPDATE project_meta SET
                            project = ?, client = ?, service = ?, page = ?, rev = ?, location = ?
                        WHERE project_id = ?
                    """, (
                        db_row.get("project"), db_row.get("client"), db_row.get("service"),
                        db_row.get("page"), db_row.get("rev"), db_row.get("location"), res[0]
                    ))
                    continue
                else:
                    next_id = con.execute("SELECT COALESCE(MAX(project_id), 0) + 1 FROM project_meta").fetchone()[0]
                    db_row["project_id"] = next_id

                    con.execute("""
                        INSERT INTO project_meta (
                            project_id, job_no, project, doc_no, item_no, client, service, page, rev, location
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        db_row["project_id"], db_row.get("job_no"), db_row.get("project"),
                        db_row.get("doc_no"), db_row.get("item_no"), db_row.get("client"),
                        db_row.get("service"), db_row.get("page"), db_row.get("rev"), db_row.get("location")
                    ))
            else:
                if project_id is None:
                    con.execute("INSERT INTO unknown (raw_json) VALUES (?)", (json.dumps(row),))
                    continue

                if table_name == "others":
                    for col, dtype in DB_TABLES[table_name].items():
                        if dtype == "BOOLEAN" and col in db_row:
                            db_row[col] = convert_value_to_type(db_row[col], "BOOLEAN")

                columns = list(db_row.keys())
                values = [db_row[c] for c in columns]
                sql_cols = ", ".join(columns)
                sql_params = ", ".join(["?" for _ in columns])
                con.execute(f"""
                    INSERT INTO {table_name} (project_id, {sql_cols}) VALUES (?, {sql_params})
                """, (project_id, *values))

        con.execute("COMMIT")
    except Exception as e:
        con.execute("ROLLBACK")
        logging.error(f"DB insert error: {e}")
        raise e
    finally:
        con.close()

def process_pdf_and_save(pdf_bytes, docname):
    tables = extract_pdf_tables_from_bytes(pdf_bytes)
    if not tables:
        raise ValueError("No tables found in PDF.")

    project_id = None
    for table in tables:
        table_type = classify_table_by_header(table)
        field_names = gpt_table_field_mapping(table, docname, table_type)
        field_names = apply_field_mapping_rules(field_names)
        rows = parse_table_to_dict(table, field_names)

        if table_type == "project_meta" and project_id is None:
            insert_table_data(table_type, None, rows, field_names)
            con = duckdb.connect(DB_DUCKDB)
            project_id = con.execute("SELECT max(project_id) FROM project_meta").fetchone()[0]
            con.close()
        else:
            if project_id is None:
                raise ValueError("Project meta must be inserted first.")
            insert_table_data(table_type, project_id, rows, field_names)
    return project_id


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H4("Controls", style={'color': 'white'}),
            dcc.Upload(id='upload', children=dbc.Button("Upload PDF", color='light'), accept='.pdf'),
            html.Br(), html.Br(),
            html.Label("Search (Keyword/Field/Table):", style={'color': 'white'}),
            dcc.Input(id='keyword', type='text', style={'width': '100%'}),
            html.Br(), dbc.Button('Auto Generate SQL', id='auto-sql', color='info', className='mt-2'),
            html.Br(), html.Br(),
            html.Label("SQL Query:", style={'color': 'white'}),
            dcc.Textarea(id='sql', style={'width': '100%', 'height': '80px'}),
            html.Br(), dbc.Button('Run Query', id='run', color='success', className='mt-2'),
        ], width=3, style={'backgroundColor': '#001f3f', 'height': '100vh', 'padding': '20px', 'borderRight': '2px solid #ccc'}),
        dbc.Col([
            html.H5('All Database Tables & Data'),
            html.Div(id='upload-result', style={'marginBottom': '20px', 'overflowY': 'scroll', 'maxHeight': '600px'}),
            html.H5('SQL Query Result'),
            html.Div(id='query-result', style={'marginBottom': '20px', 'overflowY': 'scroll', 'maxHeight': '400px'}),
        ], width=9)
    ])
], fluid=True)

@app.callback(
    Output('upload-result', 'children'),
    [Input('upload', 'contents')],
    [State('upload', 'filename')]
)
def on_upload(contents, filename):
    if not contents:
        return html.Div("Upload a PDF file to extract tables and save to database.")
    content_type, b64 = contents.split(',')
    pdf_bytes = base64.b64decode(b64)
    try:
        project_id = process_pdf_and_save(pdf_bytes, filename)
    except Exception as e:
        return html.Div(f"Error processing PDF: {e}")

    results = []
    con = duckdb.connect(DB_DUCKDB)
    tables = con.execute("SHOW TABLES").df()["name"].tolist()
    for t in tables:
        try:
            df = con.execute(f'SELECT * FROM {t}').df()
            results.append((t, df))
        except Exception:
            continue
    con.close()

    blocks = []
    for t, df in results:
        blocks.append(html.H6(f"Table: {t}"))
        if df.empty:
            blocks.append(html.Div("No data"))
        else:
            blocks.append(dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True))
    return blocks

@app.callback(
    Output('sql', 'value'),
    [Input('auto-sql', 'n_clicks')],
    [State('keyword', 'value')]
)
def fill_sql(n_clicks, keyword):
    if not n_clicks or not keyword:
        return ''
    return generate_sql_from_keywords(keyword)

def generate_sql_from_keywords(keyword):
    keyword = keyword.strip().lower()
    for table, schema in DB_TABLES.items():
        for col in schema.keys():
            if keyword in col.lower():
                return f"SELECT * FROM {table} WHERE {col} LIKE '%{keyword}%'"
    return f"SELECT * FROM project_meta WHERE project LIKE '%{keyword}%'"

@app.callback(
    Output('query-result', 'children'),
    [Input('run', 'n_clicks')],
    [State('sql', 'value')]
)
def run_query(n_clicks, sql):
    if not sql or not sql.strip():
        return html.Div("Enter your SQL query and press 'Run Query'.")
    try:
        con = duckdb.connect(DB_DUCKDB)
        result_df = con.execute(sql).df()
        con.close()
        if result_df.empty:
            return html.Div("No results.")
        return dbc.Table.from_dataframe(result_df, striped=True, bordered=True, hover=True)
    except Exception as e:
        return html.Div(f"Query error: {e}")

if __name__ == '__main__':
    init_database()
    # Render 배포용 설정
    app.run(
        host='0.0.0.0', 
        port=int(os.environ.get('PORT', 10000)),
        debug=False  # 배포시 False
    )
