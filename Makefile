prince = my_player.py
src_file = src_2147174_2117902
marsel = custom_p_divercite_heuristic.py
last_agent = abyss_v3.py

str1 = my_player.py
str2 = my_player_strategy.py

install	:
	pip install -r requirements.txt

cleandep:
	pipreqs --clean requirements.txt

getdep: 
	pipreqs --savepath requirements.txt --force .

clean:
	rm  *.json
	rm  data/*.json 
	
move:
	mv *.json data/

abyss:
	cp Divercite/${prince} ./
	mkdir ${src_file}
	cp Divercite/${src_file}/*.py ./${src_file}
	zip abyss_2147174_2117902.zip requirements.txt ${prince} ${src_file}/*.py
	rm ${prince} ${src_file} -f -R


remise:
	cp Divercite/${prince} ./
	mkdir ${src_file}
	cp Divercite/${src_file}/*.py ./${src_file}
	zip projet_2147174_2117902.zip INF8175-Rapport.pdf requirements.txt ${prince} ${src_file}/*.py
	rm ${prince} ${src_file} -f -R

challonge:
	cp Divercite/${prince} ./
	mkdir ${src_file}
	cp Divercite/${src_file}/*.py ./${src_file}
	zip LaCiteDeDieu_2147174_2117902.zip requirements.txt ${prince} ${src_file}/*.py
	rm ${prince} ${src_file} -f -R


hvp:
	python Divercite/main_divercite.py -t human_vs_computer Divercite/${prince}

hvr:
	python Divercite/main_divercite.py -t human_vs_computer Divercite/random_player_divercite.py

hvg:
	python Divercite/main_divercite.py -t human_vs_computer Divercite/greedy_player_divercite.py

gvp:
	python Divercite/main_divercite.py -t local Divercite/greedy_player_divercite.py Divercite/${prince} -r -g

pvg:
	python Divercite/main_divercite.py -t local Divercite/${prince} Divercite/greedy_player_divercite.py  -r -g

pvh:
	cd Divercite && python main_divercite.py -t human_vs_computer ${prince}

pvr:
	python Divercite/main_divercite.py -t local Divercite/${prince} Divercite/random_player_divercite.py  -r -g

rvp:
	python Divercite/main_divercite.py -t local Divercite/random_player_divercite.py  Divercite/${prince} -r -g

prince_vs_marsel:
	python Divercite/main_divercite.py -t local Divercite/${prince} Divercite/${marsel} -r -g

marsel_vs_prince:
	python Divercite/main_divercite.py -t local Divercite/${marsel} Divercite/${prince} -r -g
	
str1:
	python Divercite/main_divercite.py -t host_game -a 127.0.0.2 Divercite/${str1}

str2:
	python Divercite/main_divercite.py -t connect -a 127.0.0.2 Divercite/${str2}
	


str1_vs_str2:
	tmux new-session -d -s mysession1 "python Divercite/main_divercite.py -t host_game -a 127.0.0.1 Divercite/${str1};sleep 100"
	sleep 1
	tmux split-window -h "python Divercite/main_divercite.py -t connect -a 127.0.0.1 Divercite/${str2}"
	tmux attach-session -d
	
	


