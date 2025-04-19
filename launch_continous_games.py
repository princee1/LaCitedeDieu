import os
import glob
import subprocess
import json
import shutil
from datetime import datetime, timedelta
from threading import Timer
from time import sleep
from random import randbytes,randint

import os
# Configuration
MATCH_DURATION = 30 * 60  # 30 minutes in seconds
RESULTS_DIR = "./"
ARCHIVE_DIR = "data/"
CURRENT_ADDR = '127.0.0.10'

COUNT = 10

STATS = {}


def find_strategy_files(directory):
    return glob.glob(os.path.join(directory, "./Divercite/*_strategy.py"))

def schedule_matches(players):
    played_pairs = set()
    for i, p1 in enumerate(players):
        for p2 in players[i+1:]:
            if (p1, p2) not in played_pairs and (p2, p1) not in played_pairs:
                yield p1, p2
                played_pairs.add((p1, p2))


def run_match(player1, player2):

    process1 = subprocess.Popen(["python", f'./Divercite/main_divercite.py', '-t' ,'host_game' ,'-a', CURRENT_ADDR, f'Divercite/{player1}','-r', '-g'])
    process2 = subprocess.Popen(["python", './Divercite/main_divercite.py', '-t', 'connect', '-a', CURRENT_ADDR ,f'Divercite/${player2}'])
    
    def kill_processes():
        process1.kill()
        process2.kill()

    timeout_timer = Timer(MATCH_DURATION, kill_processes)
    timeout_timer.start()
    
    try:
        process1.wait()
        if process1.returncode != 0:
            kill_processes()
            return False
        process2.wait()
        if process2.returncode != 0:
            kill_processes()
            return False
        
    except KeyboardInterrupt:
        return True
    finally:
        timeout_timer.cancel()

    return True

def compute_stats(player1,player2):
   
    for result_file in glob.glob(os.path.join(RESULTS_DIR, "*.json")):
        with open(result_file) as f:
            data = json.load(f)
            sleep(10)
            # Update stats here (custom logic depending on JSON structure)
            scores:dict = data[-1]['scores']

            (_,scores_p1),(_,scores_p2) = scores.items()
            match_id = str(randbytes(5))
            d1 = {

                'match-id':match_id,
                'win': scores_p1 > scores_p2,
                'my_score':scores_p1,
                'opp_score':scores_p2,
                'opp_name':player2

            }

            d2 = {

                'match-id':match_id,
                'win': scores_p2 > scores_p1,
                'my_score':scores_p2,
                'opp_score':scores_p1,
                'opp_name':player1

            }
            STATS[player1]['my_total_score']+=scores_p1
            STATS[player1]['opp_total_score']+=scores_p2
            STATS[player1]['games'].append(d1)
            STATS[player1]['games_won']+= 1 if scores_p1 > scores_p2 else 0
            STATS[player1]['games_lost']+= 1 if scores_p1 < scores_p2 else 0


            STATS[player2]['my_total_score']+=scores_p2
            STATS[player2]['opp_total_score']+=scores_p1
            STATS[player2]['games'].append(d2)
            STATS[player2]['games_won']+= 1 if scores_p2 > scores_p1 else 0
            STATS[player2]['games_lost']+= 1 if scores_p2 < scores_p1 else 0


    
def archive_results(results_dir, archive_dir):
    os.makedirs(archive_dir, exist_ok=True)
    for result_file in glob.glob(os.path.join(results_dir, "*.json")):
        shutil.move(result_file, os.path.join(archive_dir, os.path.basename(result_file)))

def run_git_bash_command(command):
    """
    Runs a command in Git Bash from Python.

    Args:
        command (str): The command to run in Git Bash.

    Returns:
        str: The standard output from the command.
    """
    git_bash_path = r"C:\Program Files\Git\bin\bash.exe"  # Update if necessary
    try:
        # Run the command in Git Bash
        result = subprocess.run(
            [git_bash_path, "-c", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # Print output and error
        if result.stdout:
            print("Output:", result.stdout.strip())
        if result.stderr:
            print("Error:", result.stderr.strip())
        return result.stdout.strip()
    except FileNotFoundError:
        print("Git Bash executable not found. Check the path.")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():

    run_git_bash_command('make move')
    
    strategy_files = find_strategy_files(".")
    if not strategy_files:
        print("No strategy files found.")
        return
    print(f'Files:',strategy_files)

    for strategy_file in strategy_files:
        STATS[strategy_file] = {
            'my_total_score':0,
            'opp_total_score':0,
            'games':[],
            'games_won':0,
            'games_lost':0,

        }

    
    for _ in range(COUNT):

        for player1, player2 in schedule_matches(strategy_files):
            print(f"Running match: {player1} vs {player2}")
            success = run_match(player1, player2)
            if not success:
                print(f"Match {player1} vs {player2} failed or was terminated.")
            else:
                print(f'\tMatch {player1} vs {player2} succesfully finished')
                sleep(5)
                os.system('cls')
                compute_stats(player1,player2)
                run_git_bash_command('mv *.json data/')




        for player2, player1 in schedule_matches(strategy_files):
            print(f"Running match: {player1} vs {player2}")
            success = run_match(player1, player2)
            if not success:
                print(f"Match {player1} vs {player2} failed or was terminated.")
            else:
                print(f'\tMatch {player1} vs {player2} successfully started')
                sleep(5)
                os.system('cls')
                compute_stats(player2,player1)
                run_git_bash_command('mv *.json data/')

    
    os.system('cls')
    print("All matches completed. Results archived.")
    json.dump(STATS,open(f'simulation_{randint(0,1000000000000)}.sim.json','x'))
    run_git_bash_command('mv *sim.json sim/')

if __name__ == "__main__":
    main()
    
