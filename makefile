CXX = icpx
CXXFLAGS = -fsycl
EXE = conv

all:
	$(CXX) $(CXXFLAGS) ConvNet.cpp -o $(EXE)

run:
	./$(EXE)

clean:
	rm -rf convNet