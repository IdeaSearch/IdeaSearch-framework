
import programs.FeynmanIntegralReduction.evaluator as FeynmanIntegralReduction
import programs.SysYCompilerTest.evaluator as SysYCompilerTest
from src.FunSearch.interface import FunSearchInterface


if __name__ == "__main__":
    
    FunSearchInterface(
        program_name =  "FeynmanIntegralReduction",
        num_samplers = 10,
        num_evaluators = 10,
        max_interaction_num = 100,
    )
