from dataclasses import dataclass
import pipeline_1x1_interleaved
import pipeline_2x2_interleaved

@dataclass
class PipelineSelector(object):
    repeat_m : int
    repeat_n : int
    lds_double_buffer : bool
    interleaved : bool
    
    def pipeline_select(self):
        if self.repeat_m == 1 and self.repeat_n == 1 and self.lds_double_buffer and self.interleaved:
            return pipeline_1x1_interleaved.Pipeline1x1Interleaved()
        if self.repeat_m == 2 and self.repeat_n == 2 and self.interleaved:
            return pipeline_2x2_interleaved.Pipeline2x2Interleaved()
    
k_pipeline_1x1_lds_double_buffer_interleaved = PipelineSelector(1, 1, True, True)
k_pipeline_2x2_interleaved = PipelineSelector(2, 2, False, True)

