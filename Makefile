all:
	python setup.py build

clean:
	rm -f *.so
	rm -rf ./build
	rm -rf __pycache__

