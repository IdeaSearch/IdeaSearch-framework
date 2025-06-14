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

    # add two islands
    ideasearcher.add_island()
    ideasearcher.add_island()
    
    # Evolve for three cycles, 10 epochs on each island per cycle with ideas repopulated at the end
    for cycle in range(1, 4):
        
        ideasearcher.run(10)
        ideasearcher.repopulate_islands()
        
        best_idea = ideasearcher.get_best_idea()
        best_score = ideasearcher.get_best_score()
        print(
            f"【第{cycle}轮】"
            f"目前最高得分{best_score:.2f}，这个idea是：\n"
            f"{best_idea}\n"
        )
        
    # shutdown models
    ideasearcher.shutdown_models()


if __name__ == "__main__":
    
    main()