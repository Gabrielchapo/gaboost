all:
	python3.8 setup.py build_ext --inplace

clean:
	rm -rf *.out *.bin *.exe *.o *.a *.so test build Model/__pycache__/

re: clean all