import pandas as pd

SPE_HEIGHT = ("供試体高さ", "[cm]")
IN_T = ("in", "[s]")
OUT_T = ("out", "[s]")
INI_T = ("初期補正", "[s]")
DELTA_T = ("Δt", "[s]")
V = ("V", "[m/s]")
POISSON = ("ポアソン比", "")


def create_df(
    spe_height=None,
    in_t=None,
    out_t=None,
    ini_t=None,
    delta_t=None,
    v=None,
    poisson=None,
) -> pd.DataFrame:
    sel_df = pd.DataFrame(
        data={
            SPE_HEIGHT: spe_height,
            IN_T: in_t,
            OUT_T: out_t,
            INI_T: ini_t,
            DELTA_T: delta_t,
            V: v,
            POISSON: poisson,
        },
        index=("P", "S"),
    )
    return sel_df
