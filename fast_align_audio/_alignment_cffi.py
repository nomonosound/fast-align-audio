import os
from cffi import FFI


ffibuilder = FFI()
ffibuilder.cdef(
    "ssize_t fast_find_alignment(size_t, float *, size_t, float *, size_t, size_t);"
)

script_dir = os.path.dirname(os.path.realpath(__file__))
c_file_path = os.path.join(script_dir, "_faa.c")

with open(c_file_path, "r") as file:
    c_code = file.read()

extra_compile_args = []
if os.name == "posix":
    extra_compile_args = ["-mavx", "-Wall", "-Wextra"]

ffibuilder.set_source(
    "_fast_align_audio", c_code, extra_compile_args=extra_compile_args
)


if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
