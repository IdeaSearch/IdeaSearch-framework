from src.IdeaSearch.ideasearcher import IdeaSearcher
from programs.TemplateProgram.user_code.prompt import prologue_section as TemplateProgram_prologue_section
from programs.TemplateProgram.user_code.prompt import epilogue_section as TemplateProgram_epilogue_section
from programs.TemplateProgram.user_code.evaluation import evaluate as TemplateProgram_evaluate


def main():
    
    # pip install IdeaSearch; from IdeaSearch import IdeaSearcher (6/15)
    ideasearcher = IdeaSearcher()
    
    # load models
    ideasearcher.set_api_keys_path("src/API4LLMs/api_keys.json")
    ideasearcher.load_models()
    
    # set minimum required parameters
    ideasearcher.set_program_name("TemplateProgram")
    ideasearcher.set_database_path("programs/TemplateProgram/database")
    ideasearcher.set_evaluate_func(TemplateProgram_evaluate)
    ideasearcher.set_prologue_section(TemplateProgram_prologue_section)
    ideasearcher.set_epilogue_section(TemplateProgram_epilogue_section)
    ideasearcher.set_models([
        "Deepseek_V3",
    ])
    
    # set optional parameters
    # ideasearcher.set_assess_func(None)
    ideasearcher.set_backup_on(True)

    # add two islands
    ideasearcher.add_island()
    ideasearcher.add_island()
    
    # Evolve for three cycles, 10 epochs on each island per cycle with ideas repopulated at the end
    for _ in range(3):
        ideasearcher.run(3)
        ideasearcher.repopulate_islands()
        
    best_idea = ideasearcher.get_best_idea()
    best_score = ideasearcher.get_best_score()
    print(
        f"目前最高得分{best_score:.2f}，这个idea是：\n"
        f"{best_idea}"
    )
        
    # shutdown models (not necessary in this script, since no local models are used)
    ideasearcher.shutdown_models()


if __name__ == "__main__":
    
    main()