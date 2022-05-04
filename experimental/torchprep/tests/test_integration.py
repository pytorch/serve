from typer.testing import CliRunner
from torchprep.main import app

runner = CliRunner()



def test_resnet():
    result = runner.invoke(app, ["Camila", "--city", "Berlin"])

def test_bert():
    return NotImplemented