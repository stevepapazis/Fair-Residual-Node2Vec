all: fair_node2vec node2vec

fair_node2vec: src/fn2v_main.cpp src/fn2v.cpp
	g++ -std=c++11 -DNDEBUG -O3 -o fair_node2vec src/fn2v_main.cpp ../snap/snap-core/Snap.o src/fn2v.cpp src/n2v.cpp src/biasedrandomwalk.cpp ../snap/snap-adv/word2vec.cpp src/graph.cpp src/stochastic_random_search.cpp -I ../snap/snap-core -I ../snap/snap-adv -I ../snap/snap-exp -I ../snap/glib-core -I ../snap/glib-adv -fopenmp -lrt

# node2vec: src/n2v_main.cpp
# 	g++ -std=c++11 -DNDEBUG -O3 -o node2vec src/n2v_main.cpp ../snap/snap-core/Snap.o ../snap/snap-adv/n2v.cpp ../snap/snap-adv/biasedrandomwalk.cpp ../snap/snap-adv/word2vec.cpp -I ../snap/snap-core -I ../snap/snap-adv -I ../snap/snap-exp -I ../snap/glib-core -I ../snap/glib-adv -fopenmp -lrt
		
clean:
	rm -f *.o fair_node2vec fair_node2vec.exe node2vec node2vec.exe
