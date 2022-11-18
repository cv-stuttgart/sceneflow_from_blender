from PIL import Image
import h5py


def writeFlo5File(flow, filename):
    with h5py.File(filename, "w") as f:
        f.create_dataset("flow", data=flow, compression="gzip", compression_opts=5)


def readFlo5Flow(filename):
    with h5py.File(filename, "r") as f:
        if "flow" not in f.keys():
            raise IOError(f"File {filename} does not have a 'flow' key. Is this a valid flo5 file?")
        return f["flow"][()]


def writeDsp5File(disp, filename):
    with h5py.File(filename, "w") as f:
        f.create_dataset("disparity", data=disp, compression="gzip", compression_opts=5)


def readDsp5Disp(filename):
    with h5py.File(filename, "r") as f:
        if "disparity" not in f.keys():
            raise IOError(f"File {filename} does not have a 'disparity' key. Is this a valid dsp5 file?")
        return f["disparity"][()]


def writePngMapFile(map_, filename):
    Image.fromarray(map_).save(filename)

