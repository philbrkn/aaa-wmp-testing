import glob
import os
import sys
import time
import traceback
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from multipole_deplete_v3 import *

neutron_dir = "data/input/ENDF/ENDF-VIII-data"
endf_files = [
    i for i in glob.glob(os.path.join(neutron_dir, "*.endf")) if os.path.isfile(i)
]
out_dir = "data/output/WMP_Lib_viii.0"
# mp_file = "data/output/U238/mp_data/U238_mp-VF.pickle"
# mp_file = "data/output/U238/mp_data/U238_mp_50p_2e-3.pickle"
# mp_file = "data/output/U238/mp_data/U238_mp.pickle"
# mp_file = "data/output/U238/mp_data/U238_mp_100p_1e-3_EsigE.pickle"
mp_file = "data/output/U238/wmp_data/mp_data.pickle"
METHOD = "AAA"

print("Start processing {} nuclides - {}".format(len(endf_files), time.ctime()))


def process(endf_file):
    nuc_name = (IncidentNeutron.from_endf(endf_file)).name
    path_out = os.path.join(out_dir, nuc_name)
    if not os.path.exists(path_out):
        os.makedirs(path_out)

    # print message
    print("Processing {} - {} {} ".format(endf_file, nuc_name, time.ctime()))

    # run wmp
    time_start = time.time()
    wmp_file = os.path.join(path_out, nuc_name + ".h5")

    # if os.path.exists(wmp_file):
    #     print("Existed file for {}, processing will be skipped.".format(wmp_file))
    #     return
    try:
        # if not os.path.isfile(wmp_file):
        if os.path.isfile(mp_file):
            with open(
                os.path.join(path_out, nuc_name + "_windowing.log"),
                "w",
                buffering=1,
            ) as f:
                with redirect_stdout(f):
                    try:
                        nuc = WindowedMultipole.from_multipole(
                            mp_file,
                            search=False,
                            log=2,
                            method=METHOD,
                            n_cf=3,
                            n_win=2018,
                        )
                    except Exception as e:
                        print(f"Failed with rtol 1e-3: {str(e)}")
                        print(f"Traceback: {traceback.format_exc()}")
                        # try:
                        #     nuc = WindowedMultipole.from_multipole(mp_file, search=True, log=True, rtol=5e-3)
                        # except:
                        #     nuc = WindowedMultipole.from_multipole(mp_file, search=True, log=2, rtol=1e-2)
            # else:
            #     with open(os.path.join(path_out, nuc_name+".log"),'w') as f:
            #         with redirect_stdout(f):
            #             try:
            #                 nuc = WindowedMultipole.from_endf(endf_file,
            #                        vf_options={"log":True, "path_out":path_out},
            #                        wmp_options={"search":True, "log":True})
            #             except:
            #                 nuc = WindowedMultipole.from_endf(endf_file,
            #                        vf_options={"log":True, "rtol":5e-3, "path_out":path_out},
            #                        wmp_options={"search":True, "rtol":5e-3, "log":True})
            try:
                nuc.export_to_hdf5(wmp_file)
            except Exception:
                print(f"Traceback: {traceback.format_exc()}")
        print("Done. {} {:.1f} s".format(nuc_name, time.time() - time_start))
    except:
        print("Failed. {} {:.1f} s".format(nuc_name, time.time() - time_start))

    sys.stdout.flush()


todo_files = []
for endf_file in endf_files:
    nuc_name = (IncidentNeutron.from_endf(endf_file)).name
    if nuc_name != "U238":  # DEBUG #
        continue
    wmp_file = os.path.join(out_dir, nuc_name + ".h5")
    if os.path.isfile(wmp_file):
        # print message
        time_start = time.time()
        print("Processing {} - {} {} ".format(endf_file, nuc_name, time.ctime()))
        print("Done. {} {:.1f} s".format(nuc_name, time.time() - time_start))
    else:
        todo_files.append(endf_file)

print("{} nuclides to be processed - {}".format(len(todo_files), time.ctime()))

# with Pool(8) as p:
#    p.map(process, todo_files)
# Process nuclides sequentially since windowing is now parallelized
for endf_file in todo_files:
    process(endf_file)

print("Finish processing all nuclides - {}".format(time.ctime()))
