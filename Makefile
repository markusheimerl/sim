clean:
	rm -f *.out *.png
	$(MAKE) -C gpu clean
	$(MAKE) -C transformer clean