mkdir tempdir
javac -cp lib/selex.jar -d tempdir/ src/*/*.java
cp lib/selex.jar tempdir/NRLB.jar
(cd tempdir; jar uf NRLB.jar Jama/*)
(cd tempdir; jar uf NRLB.jar base/*)
(cd tempdir; jar uf NRLB.jar dynamicprogramming/*)
(cd tempdir; jar uf NRLB.jar minimizers/*)
(cd tempdir; jar uf NRLB.jar model/*)
(cd tempdir; jar uf NRLB.jar utils/*)
mv tempdir/NRLB.jar lib/
# rm -rf tempdir/
