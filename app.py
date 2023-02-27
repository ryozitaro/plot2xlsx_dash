import itertools as it
from pathlib import Path

import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
from dash import Dash, Input, Output, State, ctx, dcc, html
from plotly.subplots import make_subplots

import calc
from check_for_dl import JA_KEYS, KEYS, OVERLAY, WINDOW
from xlsxout import XlsxOut


class Plot:
    def __init__(self):
        self.MAX_SHEETS = 5
        self.default_spe_height = 10
        self.app = Dash(__name__, title="plot from csv to xlsx")
        self.app.layout = self.serve_layout
        # graph
        self.app.callback(
            Output("fig", "figure"),
            Output("cur_sheet", "data"),
            Output("indicator", "children"),
            Output("memory", "data"),
            Output("p_folder", "value"),
            Output("s_folder", "value"),
            Output("spe_height", "value"),
            Output("p_ini_t", "value"),
            Output("s_ini_t", "value"),
            Output("p_in_t", "children"),
            Output("p_out_t", "children"),
            Output("p_delta_t", "children"),
            Output("p_v", "children"),
            Output("s_in_t", "children"),
            Output("s_out_t", "children"),
            Output("s_delta_t", "children"),
            Output("s_v", "children"),
            Output("poi", "children"),
            Output("a1", "value"),
            Output("back", "disabled"),
            Output("forward", "disabled"),
            Output("add_sheet", "disabled"),
            Output("delete", "disabled"),
            Input("p_folder", "value"),
            Input("s_folder", "value"),
            Input("fig", "clickData"),
            Input("spe_height", "value"),
            Input("p_ini_t", "value"),
            Input("s_ini_t", "value"),
            Input("a1", "value"),
            Input("back", "n_clicks"),
            Input("forward", "n_clicks"),
            Input("add_sheet", "n_clicks"),
            Input("delete", "n_clicks"),
            State("cur_sheet", "data"),
            State("indicator", "children"),
            State("memory", "data"),
            State("fig", "figure"),
            State("p_in_t", "children"),
            State("p_out_t", "children"),
            State("p_delta_t", "children"),
            State("p_v", "children"),
            State("s_in_t", "children"),
            State("s_out_t", "children"),
            State("s_delta_t", "children"),
            State("s_v", "children"),
            State("poi", "children"),
            State("back", "disabled"),
            State("forward", "disabled"),
            State("add_sheet", "disabled"),
            State("delete", "disabled"),
            prevent_initial_call=True,
        )(self.graph)
        # download
        self.app.callback(
            Output("download_xlsx", "data"),
            Output("alert_detail", "children"),
            Output("overlay", "style"),
            Output("window", "style"),
            Input("save_button", "n_clicks"),
            Input("alert_button", "n_clicks"),
            State("save_file_name", "value"),
            State("memory", "data"),
            prevent_initial_call=True,
        )(self.download)

    def serve_layout(self):
        data_fld = tuple(
            fld.name.removeprefix("ALL")
            for fld in Path("data").glob("ALL*")
            if fld.is_dir()
        )
        return html.Div(
            children=[
                # ヘッダー
                html.Header(
                    [
                        html.Link(
                            rel="preconnect", href="https://fonts.googleapis.com"
                        ),
                        html.Link(
                            rel="preconnect",
                            href="https://fonts.gstatic.com",
                            crossOrigin="",
                        ),
                        html.Link(
                            href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP&display=swap",
                            rel="stylesheet",
                        ),
                    ],
                ),
                dcc.Store(id="cur_sheet", data=0),
                # タイトル
                html.Div(
                    children=[
                        html.H1(
                            children="plot2xlsx_dash",
                            style={"margin": "0em", "textAlign": "center"},
                        ),
                    ],
                ),
                dcc.Store(id="memory", data=[{"spe_height": self.default_spe_height}]),
                # グラフ データ入力カード
                html.Div(
                    className="card",
                    children=[
                        # グラフ列
                        html.Div(
                            children=[
                                dcc.Graph(
                                    id="fig",
                                    figure=go.Figure(),
                                    config={"displayModeBar": False},
                                    style={"width": "70vw", "height": "80vh"},
                                ),
                                html.Div(
                                    children=[
                                        html.Button(
                                            "シート消去", id="delete", disabled=True
                                        ),
                                        html.Button("＜", id="back", disabled=True),
                                        html.Span(
                                            html.Span(id="cur-indicator-shape"),
                                            id="indicator",
                                        ),
                                        html.Button("＞", id="forward", disabled=True),
                                        html.Button(
                                            "シート追加", id="add_sheet", disabled=False
                                        ),
                                    ],
                                    style={
                                        "display": "grid",
                                        "gridTemplateColumns": "2fr 2fr 1fr 2fr 2fr",
                                        "justifyItems": "center",
                                        "margin": "1em 0 0 0",
                                    },
                                ),
                            ],
                            style={"margin": "0 1vw"},
                        ),
                        # データ入力列
                        html.Div(
                            children=[
                                # ドロップダウン
                                html.Div(
                                    children=[
                                        dcc.Dropdown(
                                            options=data_fld,
                                            id="p_folder",
                                            placeholder="P波データ",
                                            clearable=False,
                                            style={"width": "8em"},
                                        ),
                                        dcc.Dropdown(
                                            options=data_fld,
                                            id="s_folder",
                                            placeholder="S波データ",
                                            clearable=False,
                                            style={"width": "8em"},
                                        ),
                                    ],
                                    style={"display": "flex", "margin": "1em 0em"},
                                ),
                                # 供試体高さ
                                html.Div(
                                    children=[
                                        "供試体高さ",
                                        dcc.Input(
                                            id="spe_height",
                                            type="number",
                                            value=self.default_spe_height,
                                            required=True,
                                            style={
                                                "width": "8em",
                                                "height": "1.5em",
                                                "margin": "0 1em",
                                            },
                                        ),
                                        html.P("[cm]"),
                                    ],
                                    style={"display": "flex", "alignItems": "center"},
                                ),
                                # P波入力
                                html.Div(
                                    children=[
                                        html.Div(
                                            className="ini_div",
                                            children=[
                                                html.P("P ini t:"),
                                                dcc.Input(
                                                    id="p_ini_t",
                                                    className="ini_input",
                                                    type="number",
                                                    placeholder="P波の初期補正値",
                                                    required=True,
                                                ),
                                                html.P("[s]"),
                                            ],
                                        ),
                                        html.P(
                                            children=(
                                                "P in t: ",
                                                html.Span(id="p_in_t"),
                                            )
                                        ),
                                        html.P(
                                            children=(
                                                "P out t: ",
                                                html.Span(id="p_out_t"),
                                            )
                                        ),
                                        html.P(
                                            children=(
                                                "P Δt: ",
                                                html.Span(id="p_delta_t"),
                                            )
                                        ),
                                        html.P(children=("P V: ", html.Span(id="p_v"))),
                                    ],
                                ),
                                # S波入力
                                html.Div(
                                    children=[
                                        html.Div(
                                            className="ini_div",
                                            children=[
                                                html.P("S ini t:"),
                                                dcc.Input(
                                                    id="s_ini_t",
                                                    className="ini_input",
                                                    type="number",
                                                    placeholder="S波の初期補正値",
                                                    required=True,
                                                ),
                                                html.P("[s]"),
                                            ],
                                        ),
                                        html.P(
                                            children=(
                                                "S in t: ",
                                                html.Span(id="s_in_t"),
                                            )
                                        ),
                                        html.P(
                                            children=(
                                                "S out t: ",
                                                html.Span(id="s_out_t"),
                                            )
                                        ),
                                        html.P(
                                            children=(
                                                "S Δt: ",
                                                html.Span(id="s_delta_t"),
                                            )
                                        ),
                                        html.P(children=("S V: ", html.Span(id="s_v"))),
                                    ],
                                ),
                                html.Br(),
                                # ポアソン比
                                html.P(
                                    children=["ポアソン比: ", html.Span(id="poi")],
                                ),
                                html.Br(),
                                # A1入力
                                html.Div(
                                    children=[
                                        "A1:",
                                        dcc.Input(
                                            id="a1",
                                            type="text",
                                            placeholder="シートのA1に入れる情報",
                                            style={
                                                "height": "1.5em",
                                                "margin": "0 1em",
                                            },
                                        ),
                                    ],
                                    style={
                                        "display": "flex",
                                        "alignItems": "center",
                                        "margin": "1em 0",
                                    },
                                ),
                            ],
                        ),
                    ],
                    style={"display": "flex"},
                ),
                html.Br(),
                # ファイル名 保存カード
                html.Div(
                    id="save_card",
                    className="card",
                    children=[
                        "ファイル名:",
                        dcc.Input(
                            id="save_file_name",
                            type="text",
                            required=True,
                            style={"height": "1.5em", "margin": "0em 0.5em"},
                        ),
                        ".xlsx",
                        dcc.Loading(
                            className="download_waiting",
                            children=[
                                html.Button(
                                    "保存", id="save_button", style={"margin": "0em 2em"}
                                ),
                                dcc.Download("download_xlsx"),
                            ],
                        ),
                    ],
                ),
                # アラート
                html.Div(
                    html.Div(
                        [
                            html.P(
                                id="alert_detail",
                                style={"fontSize": "28pt", "margin": "0em 2em"},
                            ),
                            html.Button("閉じる", id="alert_button", className="button"),
                        ],
                        id="window",
                        style={"display": "none"},
                    ),
                    id="overlay",
                    style={"display": "none"},
                ),
            ],
        )

    def read_csv(self, path: Path) -> pd.DataFrame:
        return pd.read_csv(
            path, usecols=(3, 4), index_col=0, names=["time", str(path.name)]
        )

    def folder_select(self, num):
        csv1 = Path(f"data/ALL{num}/F{num}CH1.CSV")
        csv2 = csv1.with_stem(f"F{num}CH2")

        data: dict[Path, pd.DataFrame] = {}

        for path in (csv1, csv2):
            data[path] = self.read_csv(path)
        df = data[csv1].join(data[csv2])
        return df

    def new_plot(self, p_folder, s_folder):
        p_df = self.folder_select(p_folder)
        s_df = self.folder_select(s_folder)
        ps_df = p_df.join(s_df)
        titles = ps_df.columns

        fig = make_subplots(
            rows=2,
            cols=2,
            horizontal_spacing=0.06,
            vertical_spacing=0.1,
            subplot_titles=titles,
        )

        for col_index, (row, col) in zip(titles, it.product(range(1, 3), range(1, 3))):
            fig.add_trace(
                go.Scatter(x=ps_df.index, y=ps_df[col_index], line_color="blue"),
                row,
                col,
            )
            fig.add_vline(None, row, col, line_width=2, line_color="red", visible=False)
        fig.update_layout(
            showlegend=False,
            hovermode="x",
            clickmode="event",
            margin=dict(t=20, b=10, l=10, r=10),
            plot_bgcolor="#fff",
        )
        fig.update_traces(hoverinfo="x")
        fig.update_xaxes(
            zeroline=False,
            linecolor="black",
            mirror="allticks",
            gridcolor="#d0d0d0",
            spikemode="across",
            spikecolor="gray",
            spikethickness=2,
        )
        fig.update_yaxes(
            zeroline=False, linecolor="black", mirror="allticks", gridcolor="#d0d0d0"
        )

        return fig, p_df, s_df

    def graph(
        self,
        p_folder,
        s_folder,
        click,
        spe_height,
        p_ini_t,
        s_ini_t,
        a1,
        backclick,
        forwardclick,
        add_sheetclick,
        deleteclick,
        cur_sheet: int,
        indicator,
        memory: list[dict],
        fig,
        p_in_t,
        p_out_t,
        p_delta_t,
        p_v,
        s_in_t,
        s_out_t,
        s_delta_t,
        s_v,
        poi,
        backdis,
        fowarddis,
        add_sheetdis,
        deletedis,
    ):

        trg = ctx.triggered_id
        if trg in {"p_folder", "s_folder"}:
            if not None in {p_folder, s_folder}:
                p_in_t = (
                    p_out_t
                ) = p_delta_t = p_v = s_in_t = s_out_t = s_delta_t = s_v = poi = None

                fig, p_df, s_df = self.new_plot(p_folder, s_folder)

                memory[cur_sheet] |= {
                    "fig": fig.to_json(),
                    "p_folder": p_folder,
                    "s_folder": s_folder,
                    "p_df": p_df.to_dict(),
                    "s_df": s_df.to_dict(),
                }

        elif trg in {"fig", "spe_height", "p_ini_t", "s_ini_t"}:
            if "fig" in memory[cur_sheet]:
                fig = pio.from_json(memory[cur_sheet]["fig"])
                p_in_t = fig.layout.shapes[0].x0
                p_out_t = fig.layout.shapes[1].x0
                s_in_t = fig.layout.shapes[2].x0
                s_out_t = fig.layout.shapes[3].x0
                if trg == "fig":
                    curvenumber = click["points"][0]["curveNumber"]
                    x = click["points"][0]["x"]
                    fig.layout.shapes[curvenumber].update(
                        dict(
                            x0=x,
                            x1=x,
                            visible=True,
                        )
                    )
                    match curvenumber:
                        case 0:
                            p_in_t = x
                        case 1:
                            p_out_t = x
                        case 2:
                            s_in_t = x
                        case 3:
                            s_out_t = x

                p_delta_t = calc.delta(p_out_t, p_in_t, p_ini_t)
                p_v = calc.v(spe_height, p_delta_t)
                s_delta_t = calc.delta(s_out_t, s_in_t, s_ini_t)
                s_v = calc.v(spe_height, s_delta_t)

                poi = calc.poisson(p_v, s_v)

                memory[cur_sheet] |= {
                    "fig": fig.to_json(),
                    "spe_height": spe_height,
                    "p_ini_t": p_ini_t,
                    "s_ini_t": s_ini_t,
                    "p_in_t": p_in_t,
                    "s_in_t": s_in_t,
                    "p_out_t": p_out_t,
                    "s_out_t": s_out_t,
                    "p_delta_t": p_delta_t,
                    "s_delta_t": s_delta_t,
                    "p_v": p_v,
                    "s_v": s_v,
                    "poi": poi,
                }

                p_in_t = self.make_prefix(p_in_t)
                p_out_t = self.make_prefix(p_out_t)
                p_delta_t = self.make_prefix(p_delta_t)
                if p_v is not None:
                    p_v = f"{p_v} [m/s]"

                s_in_t = self.make_prefix(s_in_t)
                s_out_t = self.make_prefix(s_out_t)
                s_delta_t = self.make_prefix(s_delta_t)
                if s_v is not None:
                    s_v = f"{s_v} [m/s]"

            else:
                memory[cur_sheet] |= {
                    "spe_height": spe_height,
                    "p_ini_t": p_ini_t,
                    "s_ini_t": s_ini_t,
                }

        elif trg == "a1":
            memory[cur_sheet] |= {"a1": a1}

        elif trg in {"back", "forward", "add_sheet", "delete"}:
            if trg == "add_sheet":
                cur_sheet += 1
                memory.insert(cur_sheet, {"spe_height": self.default_spe_height})
                p_folder = s_folder = None
                fig = go.Figure()
                spe_height = self.default_spe_height
                p_ini_t = p_in_t = p_out_t = p_delta_t = p_v = None
                s_ini_t = s_in_t = s_out_t = s_delta_t = s_v = None
                poi = a1 = None

            else:
                flag = False
                if trg == "delete":
                    del memory[cur_sheet]
                    if cur_sheet == len(memory):
                        cur_sheet -= 1
                        flag = True
                elif trg == "back":
                    cur_sheet -= 1
                    flag = True
                elif trg == "forward":
                    cur_sheet += 1
                    flag = True
                if flag:
                    cur_mem = memory[cur_sheet]
                    fig = (
                        pio.from_json(cur_mem["fig"])
                        if cur_mem.get("fig")
                        else go.Figure()
                    )
                    spe_height = cur_mem.get("spe_height")
                    p_folder = cur_mem.get("p_folder")
                    s_folder = cur_mem.get("s_folder")
                    p_ini_t = cur_mem.get("p_ini_t")
                    s_ini_t = cur_mem.get("s_ini_t")
                    p_in_t = cur_mem.get("p_in_t")
                    s_in_t = cur_mem.get("s_in_t")
                    p_out_t = cur_mem.get("p_out_t")
                    s_out_t = cur_mem.get("s_out_t")
                    p_delta_t = cur_mem.get("p_delta_t")
                    s_delta_t = cur_mem.get("s_delta_t")
                    p_v = cur_mem.get("p_v")
                    s_v = cur_mem.get("s_v")
                    poi = cur_mem.get("poi")
                    a1 = cur_mem.get("a1")

            indicator = [
                html.Span(className="indicator-shape", id="cur-indicator-shape")
                if m == cur_sheet
                else html.Span(className="indicator-shape")
                for m in range(len(memory))
            ]
            backdis = True if cur_sheet == 0 else False
            fowarddis = True if cur_sheet + 1 == len(memory) else False
            add_sheetdis = True if len(memory) == self.MAX_SHEETS else False
            deletedis = True if len(memory) == 1 else False
        return (
            fig,
            cur_sheet,
            indicator,
            memory,
            p_folder,
            s_folder,
            spe_height,
            p_ini_t,
            s_ini_t,
            p_in_t,
            p_out_t,
            p_delta_t,
            p_v,
            s_in_t,
            s_out_t,
            s_delta_t,
            s_v,
            poi,
            a1,
            backdis,
            fowarddis,
            add_sheetdis,
            deletedis,
        )

    def make_prefix(self, num, unit="sec"):
        if isinstance(num, int | float):
            if (abs_num := abs(num)) >= 1:
                return f"{num} [{unit}]"
            else:
                for p in ((1e-3, "m"), (1e-6, "μ"), (1e-9, "n"), (1e-12, "p")):
                    if abs_num < p[0] * 1000:
                        u = p
                return f"{num/u[0]:f} [{u[1]}{unit}]"
        return num

    def download(self, save_button, alert_button, file_name, memory: list[dict]):
        def export_xlsx(xlsx):
            with XlsxOut(xlsx) as xo:
                for sheet in memory:
                    xo.main_write(sheet)

        trg = ctx.triggered_id
        if trg == "alert_button":
            return None, None, {"display": "none"}, {"display": "none"}

        if not file_name:
            return None, "ファイル名を入れてください", OVERLAY, WINDOW
        for i, di in enumerate(memory, start=1):
            for k in KEYS:
                if di.get(k) is None:
                    return None, f"{i}番目のシート\n{JA_KEYS[k]}が未入力です。", OVERLAY, WINDOW

        return (
            dcc.send_bytes(export_xlsx, f"{file_name}.xlsx"),
            None,
            {"display": "none"},
            {"display": "none"},
        )


plot = Plot()

plot.app.run_server(debug=True)
