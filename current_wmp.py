import h5py
# with h5py.File("/home/philip/Research/WMP_Library/092238.h5", "r") as f:
with h5py.File("/home/philip/Research/wmp_testing/U238-vf-wmp-VIII.h5", "r") as f:
    # # List all options (groups and datasets) in the HDF5 file
    # def list_all(name, obj):
    #     print(name, "(Group)" if isinstance(obj, h5py.Group) else "(Dataset)")
    # f.visititems(list_all)

    # Find names
    def walk(name, obj):
        if isinstance(obj, h5py.Dataset) and "pole" in name.lower():
            print(name, obj.shape)
    f.visititems(walk)
    
    # expand data:
    def expand(name, obj):
        if isinstance(obj, h5py.Dataset) and "data" in name.lower():
            print(name, obj.shape)
    f.visititems(expand)
    
    # Print the number of inner windows
    windows = f["U238/windows"]
    n_windows = windows.shape[0]
    print("Number of inner windows:", n_windows)
    poles = f["U238/data"]
    n_poles = poles.shape[0]
    print("Number of poles:", n_poles)