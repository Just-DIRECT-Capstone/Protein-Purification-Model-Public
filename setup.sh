echo "starting setup..."
echo " "

echo "making 'JUST' conda virtual environment..."
conda env create -f environment.yml
echo " "

echo "running tests"
call activate JUST
cd tests
coverage run -m unittest test.py
echo " "

echo "If errors, please open an issue on our github. https://github.com/FermentAI/Reactor-Modeling-Sandbox/issues"
echo " "

echo " all done!"
