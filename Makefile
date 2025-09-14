clean:
	rm -f *.out *.bin *.png
	$(MAKE) -C gpu clean
	$(MAKE) -C transformer clean