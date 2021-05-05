echo "starting setup..."
echo " "

echo "making 'JUST' conda virtual environment..."
conda env create -f environment.yml
pip install -r requirements.txt
echo " "

echo "running tests"
call activate JUST
coverage run -m unittest tests/test.py
echo " "

echo "If errors, please open an issue on our github. https://https://github.com/Just-DIRECT-Capstone/Protein-Purification-Model-Public/issues"
echo " "

echo " all done!"
