import io
from pathlib import Path

import pandas as pd
import plotly.io as pio
import xlsxwriter as xw
from xlsxwriter.chart_scatter import ChartScatter
from xlsxwriter.utility import xl_cell_to_rowcol, xl_rowcol_to_cell

import calc
import sel_idx

SEL_DF_ROW_NUM = {"P": 2, "S": 3}


class XlsxOut:
    def __init__(self, xlsx) -> None:
        self.xlsx = xlsx
        self.wb = xw.Workbook(self.xlsx)

    def close(self):
        self.wb.close()

    def __enter__(self):
        return self

    def __exit__(self, ex_type, ex_value, traceback):
        self.close()

    def _cell_judge(self, cell: str | tuple[int, int]) -> tuple[int, int]:
        """
        rowとcolのタプルによるセル指定か、strによるセル指定か判定します。
        """
        if isinstance(cell, str):
            row, col = xl_cell_to_rowcol(cell)
        else:
            row, col = cell

        return row, col

    def _write_file_data(
        self, cell: str | tuple[int, int], p_or_s: str, df: pd.DataFrame
    ) -> None:
        """
        入力したcsvデータをファイル名とともにxlsxに貼り付けます。
        """
        row, col = self._cell_judge(cell)

        path_1, path_2 = df.columns

        # ファイル列書き出し
        self.ws.write_column(row, col, ["ファイル名", path_1, path_2])

        df = df.reset_index()
        df.columns = [f"{p_or_s}波時間", f"{p_or_s}波CH1", f"{p_or_s}波CH2"]

        # データ列たち書き出し
        for i in df.items():
            col += 1
            self.ws.write_column(row, col, [i[0], *i[1]])

    def _create_chart(
        self, cell: str | tuple[int, int], data_length: int
    ) -> ChartScatter:
        """
        チャートオブジェクトを作成します。
        """
        row, col = self._cell_judge(cell)
        data_length -= 1
        chart = self.wb.add_chart({"type": "scatter", "subtype": "smooth"})
        chart.set_size({"x_scale": 1.2, "y_scale": 1.1})
        chart.add_series(
            {
                "name": "out",
                "categories": [self.ws.get_name(), row, col, row + data_length, col],
                "values": [
                    self.ws.get_name(),
                    row,
                    col + 2,
                    row + data_length,
                    col + 2,
                ],
            }
        )
        chart.add_series(
            {
                "name": "in",
                "categories": [self.ws.get_name(), row, col, row + data_length, col],
                "values": [
                    self.ws.get_name(),
                    row,
                    col + 1,
                    row + data_length,
                    col + 1,
                ],
                "y2_axis": True,
            }
        )
        chart.set_x_axis({"crossing": "max", "major_gridlines": {"visible": True}})
        chart.set_x2_axis({"crossing": "min"})
        chart.set_y_axis({"crossing": "min", "major_gridlines": {"visible": True}})
        return chart

    def _locate_cells(self, series: pd.Series, row: int, col: int) -> pd.Series:
        col += 1
        return pd.Series(
            xl_rowcol_to_cell(row + SEL_DF_ROW_NUM[series.name], col + i)
            for i in range(len(series))
        )

    def _cell_contents(
        self, cell_idx: pd.DataFrame, border, name, p_num, s_num
    ) -> tuple[list, list]:
        match name:
            case sel_idx.DELTA_T:
                cols = [sel_idx.OUT_T, sel_idx.IN_T, sel_idx.INI_T]
                p_fml = calc.delta(*cell_idx.loc["P", cols], formula=True)
                s_fml = calc.delta(*cell_idx.loc["S", cols], formula=True)
            case sel_idx.V:
                cols = [sel_idx.SPE_HEIGHT, sel_idx.DELTA_T]
                p_fml = calc.v(*cell_idx.loc["P", cols], formula=True)
                s_fml = calc.v(*cell_idx.loc["S", cols], formula=True)
            case sel_idx.POISSON:
                p_fml = s_fml = calc.poisson(*cell_idx[sel_idx.V], formula=True)
            case _:
                p_fml = s_fml = None

        if p_fml:
            p_cell_contents = [p_fml, border, p_num]
            s_cell_contents = [s_fml, border, s_num]
        else:
            p_cell_contents = [p_num, border]
            s_cell_contents = [s_num, border]
        return p_cell_contents, s_cell_contents

    def _write_sel_df(self, cell: str | tuple[int, int], sel_df: pd.DataFrame) -> None:
        """
        dfを書き込み用の形に直し、書き込まれるセルを特定したDataFrameを作り、
        選択データをxlsxに書き込みます。
        """
        row, col = self._cell_judge(cell)

        # 書き込みデータの用意
        data = sel_df.reset_index(names="").items()

        # 書き込まれるセルの特定
        cell_idx = sel_df.copy()
        cell_idx[:] = ""
        cell_idx = cell_idx.apply(
            lambda x: self._locate_cells(x, row, col), axis=1, result_type="broadcast"
        )

        # セルのボーダー
        border_btm_empty = self.wb.add_format(
            {"border": 1, "bottom": 0, "font_size": 9}
        )
        border_top_empty = self.wb.add_format({"border": 1, "top": 0})
        border = self.wb.add_format({"border": 1})
        border_font_size_9 = self.wb.add_format({"border": 1, "font_size": 9})

        # 書き込み
        for name, (p_num, s_num) in data:
            # ヘッダー書き込み
            if name == sel_idx.POISSON:
                self.ws.merge_range(row, col, row + 1, col, name[0], border_font_size_9)
                self.ws.merge_range(row + 2, col, row + 3, col, None)
            else:
                self.ws.write(row, col, name[0], border_btm_empty)
                self.ws.write(row + 1, col, name[1], border_top_empty)

            # 値書き込み
            p_cell_contents, s_cell_contents = self._cell_contents(
                cell_idx, border, name, p_num, s_num
            )
            self.ws.write(row + 2, col, *p_cell_contents)
            self.ws.write(row + 3, col, *s_cell_contents)

            col += 1

    def _expand_sheet(
        self, sheet
    ) -> tuple[io.BytesIO, pd.DataFrame, pd.DataFrame, pd.DataFrame, Path, Path]:
        fig = pio.from_json(sheet["fig"])
        out_img = io.BytesIO(fig.to_image(format="png", width=1440, height=960))
        sel_df = sel_idx.create_df(
            sheet["spe_height"],
            [sheet["p_in_t"], sheet["s_in_t"]],
            [sheet["p_out_t"], sheet["s_out_t"]],
            [sheet["p_ini_t"], sheet["s_ini_t"]],
            [sheet["p_delta_t"], sheet["s_delta_t"]],
            [sheet["p_v"], sheet["s_v"]],
            sheet["poi"],
        )
        p_df = pd.DataFrame(sheet["p_df"])
        s_df = pd.DataFrame(sheet["s_df"])
        bmp_path = lambda x: Path(f"data/ALL{x}/F{x}TEK.BMP")
        p_bmp = bmp_path(sheet["p_folder"])
        s_bmp = bmp_path(sheet["s_folder"])
        return out_img, sel_df, p_df, s_df, p_bmp, s_bmp

    def main_write(self, sheet: dict) -> None:
        out_img, sel_df, p_df, s_df, p_bmp, s_bmp = self._expand_sheet(sheet)

        self.ws = self.wb.add_worksheet()
        ws = self.ws

        ws.write("A1", sheet["a1"])

        ws.write("A2", "P波")
        ws.insert_image("A3", p_bmp, {"x_scale": 0.8, "y_scale": 0.8})
        ws.write("E2", "P波")
        ws.insert_image("E3", s_bmp, {"x_scale": 0.8, "y_scale": 0.8})

        self._write_file_data("J1", "P", p_df)
        self._write_file_data("N1", "S", s_df)

        ws.write("A16", "P波")
        chart = self._create_chart("K2", len(p_df))
        ws.insert_chart("A17", chart)

        ws.write("A35", "S波")
        chart = self._create_chart("O2", len(s_df))
        ws.insert_chart("A36", chart)

        self._write_sel_df("B58", sel_df)

        ws.insert_image(
            "A64", "", {"image_data": out_img, "x_scale": 0.3, "y_scale": 0.3}
        )
