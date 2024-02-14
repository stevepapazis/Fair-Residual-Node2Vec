include ../snap/Makefile.config

MAIN := node2vec
SRC  := src
DEPH := $(SRC)/n2v.h $(SRC)/word2vec.h $(SRC)/biasedrandomwalk.h
DEPCPP = $(SRC)/n2v.cpp $(SRC)/word2vec.cpp $(SRC)/biasedrandomwalk.cpp
CXXFLAGS += $(CXXOPENMP)

SNAPPATH  := ../snap
EXSNAP    := $(SNAPPATH)/snap-core
EXSNAPADV := $(SNAPPATH)/snap-adv
EXGLIB	  := $(SNAPPATH)/glib-core
EXSNAPEXP := $(SNAPPATH)/snap-exp


all: $(MAIN)

$(MAIN): $(SRC)/$(MAIN).cpp $(DEPH) $(DEPCPP) $(EXSNAP)/Snap.o 
	$(CC) $(CXXFLAGS) -o $(MAIN) $(SRC)/$(MAIN).cpp $(DEPCPP) $(EXSNAP)/Snap.o -I$(EXSNAP) -I$(EXSNAPADV) -I$(EXGLIB) -I$(EXSNAPEXP) $(LDFLAGS) $(LIBS)

$(EXSNAP)/Snap.o: 
	make -C $(EXSNAP)

clean:
	rm -f *.o  $(MAIN)  $(MAIN).exe


