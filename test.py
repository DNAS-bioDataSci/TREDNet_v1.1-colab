import os, sys
p = "/net/intdev/devdcode/gaetano/projects/TREDNet_v1.1/v0.1/phase_one_model.h5"
print("exists:", os.path.exists(p), "size:", os.path.getsize(p))
with open(p, "rb") as f:
    head = f.read(8)
print("header:", head)  # should be b'\x89HDF\r\n\x1a\n'TREDNet_v1.1