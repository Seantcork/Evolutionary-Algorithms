CC = g++
CFLAGS = -Wall

evAlg: evAlt.cpp
	 $(CC) $(CFLAGS) -o $@ evAlt.cpp

clean:
	 rm -f evAlg