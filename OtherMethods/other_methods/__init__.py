from .clever import CLEVER
from .fast_lip import FastLip
from .naive_methods import NaiveUB, RandomLB
from .seq_lip import SeqLip 
from .other_methods import OtherResult

OTHER_METHODS = [CLEVER, FastLip, NaiveUB, RandomLB, SeqLip]#, LipSDP]
LOCAL_METHODS = [CLEVER, FastLip, RandomLB]
GLOBAL_METHODS = [NaiveUB, SeqLip]#, LipSDP