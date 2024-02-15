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

    def gen_args(self):
        args_str = ".args: \n"
        offset = 0
        for key in self.args.keys():
            ka_trait = self.agrs[key]
            ele_str = "- \{ .name {}, .size: {}, .offset: {}, .value_kind: {}, .value_type: {}".format(
                      key,
                      ka_trait.size,
                      offset,
                      ka_trait.value_kind)
            offset += ka_trait.size
            if ka_trait.address_space == global_buffer : 
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
    .reqd_workgroup_size : [{}, {}, {}]
    .max_flat_workgroup_size: {}
    .args:
    - { .name: ptr_c,           .size: 8, .offset:   0, .value_kind: global_buffer, .value_type: f16, .address_space: global, .is_const: false}
    - { .name: ptr_a,           .size: 8, .offset:   8, .value_kind: global_buffer, .value_type: f16, .address_space: global, .is_const: true }
    - { .name: ptr_b,           .size: 8, .offset:  16, .value_kind: global_buffer, .value_type: f16, .address_space: global, .is_const: true }
    - { .name: ptr_scale,       .size: 8, .offset:  24, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: true}
    - { .name: m,               .size: 4, .offset:  32, .value_kind: by_value, .value_type: i32}
    - { .name: n,               .size: 4, .offset:  36, .value_kind: by_value, .value_type: i32}
    - { .name: k,               .size: 4, .offset:  40, .value_kind: by_value, .value_type: i32}
    - { .name: lda,             .size: 4, .offset:  44, .value_kind: by_value, .value_type: i32}
    - { .name: ldb,             .size: 4, .offset:  48, .value_kind: by_value, .value_type: i32}
    - { .name: ldc,             .size: 4, .offset:  52, .value_kind: by_value, .value_type: i32}
    - { .name: k_per_cta,       .size: 4, .offset:  56, .value_kind: by_value, .value_type: i32}
    - { .name: print,           .size: 8, .offset:  60, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: false}
...
.end_amdgpu_metadata
""" 
