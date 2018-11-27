# nanogenmo-2018
Technique is the typical RNN generator approach where a recurrent neural network repeatedly predicts the next word. It is trained on the complete CreepyPasta corpus as of October, 2018. Out of respect for the website, I have not included the corpus itself here. Instead, the data directory contains some classic horror along with war and peace so that others can train their own models if they like.


# CreepyPasta Generator
usage: generate.py [-h] [--length WORDS] [--paragraph_mean_length WORDS_IN_PAR]
                   [--paragraph_standard_deviation STD_DEV_PAR_LENGTH] [--start_crazy START_CRAZY] [--end_crazy END_CRAZY]
                   [--weights_dir WEIGHTS_DIR] [--verbose VERBOSE]

Generate a CreepyPasta story.

optional arguments:
  -h, --help            show this help message and exit
  --length WORDS        how many words?
  --paragraph_mean_length WORDS_IN_PAR
                        roughly how long should a paragraph be?
  --paragraph_standard_deviation STD_DEV_PAR_LENGTH
                        how much should paragraph length vary? (between 0 and 1)
  --start_crazy START_CRAZY
                        how crazy should it start?
  --end_crazy END_CRAZY
                        how crazy should it end?
  --weights_dir WEIGHTS_DIR
                        where are the weights?
  --verbose VERBOSE     print the story to the console (it's always written to mynovel.txt)