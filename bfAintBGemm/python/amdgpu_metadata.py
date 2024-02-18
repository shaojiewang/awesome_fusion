from dataclasses import dataclass

@dataclass
class AmdgpuMetadata(object):
    hsa_version : list
    name : str
    sgpr_count : int
    vgpr_count : int
    kernarg_segment_align : int 
    kernarg_segment_size : int
    group_segment_fixed_size : int 
    private_segment_fixed_size : int
    wavefront_size : int
    reqd_workgroup_size : list
    max_flat_workgroup_size : int
    args : dict

    def __post_init__(self):
        args_str = self.gen_args()
        k_metadata_body = self.write_metadata()
        self.metadata_body = k_metadata_body.format(
            self.hsa_version[0], self.hsa_version[1],
            self.name,
            self.name,
            self.sgpr_count,
            self.vgpr_count,
            self.kernarg_segment_align,
            self.kernarg_segment_size,
            self.group_segment_fixed_size,
            self.private_segment_fixed_size,
            self.wavefront_size,
            self.reqd_workgroup_size[0], self.reqd_workgroup_size[1], self.reqd_workgroup_size[2],
            self.max_flat_workgroup_size,
            args_str)

    def gen_args(self):
        args_str = "\n"
        offset = 0
        for key in self.args.keys():
            ka_trait = self.args[key]
            ele_str = "      - {{ .name {}, .size: {}, .offset: {}, .value_kind: {}, .value_type: {}".format(
                      key,
                      ka_trait.size,
                      offset,
                      ka_trait.value_kind,
                      ka_trait.value_type)
            offset += ka_trait.size
            if ka_trait.address_space == 'global' : 
                ele_str += ", .address_space: {}, .is_const: {}".format(
                           ka_trait.address_space,
                           "true" if ka_trait.is_const else "false")
            ele_str += "} \n"
            args_str += ele_str
        return args_str
            
            
    def write_metadata(self):
        METADATA = """
.amdgpu_metadata
---
amdhsa.version: [ {}, {} ]
amdhsa.kernels:
  - .name: {}
    .symbol: {}.kd
    .sgpr_count: {}
    .vgpr_count: {}
    .kernarg_segment_align: {}
    .kernarg_segment_size: {}
    .group_segment_fixed_size: {}
    .private_segment_fixed_size: {}
    .wavefront_size: {}
    .reqd_workgroup_size: [{}, {}, {}]
    .max_flat_workgroup_size: {}
    .args: {}
...
.end_amdgpu_metadata
""" 
        return METADATA
        
