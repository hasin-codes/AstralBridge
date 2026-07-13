from astralbridge.gemma import StaticGemmaClient
from astralbridge.interpretation import ScientificInterpreter


def test_interpreter_uses_gemma_client():
    client = StaticGemmaClient(responses=["Predicted redshift z=0.341 from image tokens."])
    interpreter = ScientificInterpreter(client)

    text = interpreter.interpret(
        prediction_value="z=0.341",
        target="tok_z",
        modalities={"tok_image": 576, "tok_flux_g": 1},
        lineage={"survey": "Roman WFI demo"},
    )

    assert "z=0.341" in text
    assert client.calls
