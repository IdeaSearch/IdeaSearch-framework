# pip install IdeaSearch; from IdeaSearch import IdeaSearcher (6/15)
from src.IdeaSearch.ideasearcher import IdeaSearcher
from programs.TemplateProgram.user_code.prompt import system_prompt as TemplateProgram_system_prompt
from programs.TemplateProgram.user_code.prompt import prologue_section as TemplateProgram_prologue_section
from programs.TemplateProgram.user_code.prompt import epilogue_section as TemplateProgram_epilogue_section
from programs.TemplateProgram.user_code.evaluation import evaluate as TemplateProgram_evaluate
from programs.TemplateProgram.user_code.mutation import mutate as TemplateProgram_mutate
from programs.TemplateProgram.user_code.crossover import crossover as TemplateProgram_crossover
from programs.TemplateProgram.user_code.assessment import assess as TemplateProgram_assess


def main():
    
    ideasearcher = IdeaSearcher()
    
    ideasearcher.set_api_keys_path("src/API4LLMs/api_keys.json")
    ideasearcher.set_local_models_path(None)
    ideasearcher.load_models()
    
    ideasearcher.set_program_name("TemplateProgram")
    ideasearcher.set_evaluate_func(TemplateProgram_evaluate)
    ideasearcher.set_system_prompt(TemplateProgram_system_prompt)
    ideasearcher.set_prologue_section(TemplateProgram_prologue_section)
    ideasearcher.set_epilogue_section(TemplateProgram_epilogue_section)
    
    ideasearcher.set_mutation_func([
        None,
        TemplateProgram_mutate,
    ][0])
    ideasearcher.set_crossover_func([
        None,
        TemplateProgram_crossover,
    ][0])
    ideasearcher.set_assess_func(TemplateProgram_assess)
    
    for epoch in range(1, 11):
        
        ideasearcher.set_mutation_num(epoch)
        ideasearcher.set_crossover_num(epoch * 2)
        
        ideasearcher.run(10)
    



if __name__ == "__main__":
    
    main()