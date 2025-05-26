# pip install IdeaSearch; from IdeaSearch import IdeaSearcher (6/15)
from src.IdeaSearch.ideasearcher import IdeaSearcher
from src.utils import clear_file_content
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
    ideasearcher.set_database_path("programs/TemplateProgram/database/")
    ideasearcher.set_evaluate_func(TemplateProgram_evaluate)
    ideasearcher.set_system_prompt(TemplateProgram_system_prompt)
    ideasearcher.set_prologue_section(TemplateProgram_prologue_section)
    ideasearcher.set_epilogue_section(TemplateProgram_epilogue_section)
    
    ideasearcher.set_generate_num(1)
    
    ideasearcher.set_models([
        "Deepseek_V3",
    ])
    ideasearcher.set_model_temperatures([
        1.0,
    ])
    
    ideasearcher.set_mutation_func([
        None,
        TemplateProgram_mutate,
    ][0])
    ideasearcher.set_crossover_func([
        None,
        TemplateProgram_crossover,
    ][0])
    ideasearcher.set_assess_func([
        None,
        TemplateProgram_assess,
    ][0])
    ideasearcher.set_assess_interval(1)
    
    ideasearcher.set_generation_bonus(2.0)

    diary_path = "programs/TemplateProgram/database/log/diary.txt"
    clear_file_content(diary_path)
    ideasearcher.set_diary_path(diary_path)

    # add two islands
    ideasearcher.add_island()
    ideasearcher.add_island()
    
    for epoch in range(1, 4):
        
        ideasearcher.set_mutation_num(epoch)
        ideasearcher.set_crossover_num(epoch * 2)
        
        ideasearcher.run(5)
        ideasearcher.repopulate_islands()
        
    ideasearcher.shutdown_models()


if __name__ == "__main__":
    
    main()