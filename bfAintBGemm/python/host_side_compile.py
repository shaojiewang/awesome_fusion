import subprocess
import os

class HostSideCompile(object):
    def __init__(self, code_path, exe_path, tmp_path = '/tmp', arch = "90a"):
        self.host_main_code_path = code_path
        self.exe_path = exe_path
        self.tmp_path = tmp_path
        self.arch = arch

    def compile_host(self):
        if os.path.exists(self.host_main_code_path):
            compile_cmd = ['/opt/rocm/bin/hipcc']
            compile_cmd.append(self.host_main_code_path)
            compile_cmd.append('-fPIC')
            compile_cmd.append('-std=c++17')
            compile_cmd.append('-O3')
            compile_cmd.append('-Wall')
            # compile_cmd.append('-DASM_PRINT')
            compile_cmd.append(f'--offload-arch=gfx{self.arch}')
            # compile_cmd.append('-save-temps={}'.format(self.tmp_path)) # TODO: let temp file stored in the right path
            compile_cmd.append('-o')
            compile_cmd.append(self.exe_path)
            # print(compile_cmd)
            subprocess.run(compile_cmd, stdout=subprocess.PIPE)
        else:
            assert false, "{} file is not generated yet".format(self.host_main_code_path)


