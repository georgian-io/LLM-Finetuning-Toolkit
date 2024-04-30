from unittest.mock import MagicMock

from transformers import BitsAndBytesConfig

from llmtune.finetune.lora import LoRAFinetune
from test_utils.test_config import get_sample_config


def test_lora_finetune_initialization(mocker):
    """Test the initialization of LoRAFinetune with a sample configuration."""
    # Mock dependencies that LoRAFinetune might call during initialization
    mocker.patch("llmtune.finetune.lora.AutoModelForCausalLM.from_pretrained")
    mocker.patch("llmtune.finetune.lora.AutoTokenizer.from_pretrained")
    mock_lora_config = mocker.patch("llmtune.finetune.lora.LoraConfig")
    mocker.patch(
        "llmtune.finetune.lora.LoRAFinetune._inject_lora",
        return_value=None,  # _inject_lora doesn't return a value
    )

    # Initialize LoRAFinetune with the sample configuration
    lora_finetune = LoRAFinetune(
        config=get_sample_config(), directory_helper=MagicMock()
    )
    # Assertions to ensure that LoRAFinetune is initialized as expected
    mock_lora_config.assert_called_once_with(**get_sample_config().lora.model_dump())

    assert (
        lora_finetune.config == get_sample_config()
    ), "Configuration should match the input configuration"


def test_model_and_tokenizer_loading(mocker):
    # Prepare the configuration
    sample_config = get_sample_config()

    mock_model = mocker.patch(
        "llmtune.finetune.lora.AutoModelForCausalLM.from_pretrained",
        return_value=MagicMock(),
    )
    mock_tokenizer = mocker.patch(
        "llmtune.finetune.lora.AutoTokenizer.from_pretrained", return_value=MagicMock()
    )
    mock_inject_lora = mocker.patch(
        "llmtune.finetune.lora.LoRAFinetune._inject_lora",
        return_value=None,  # _inject_lora doesn't return a value
    )
    directory_helper = MagicMock()
    LoRAFinetune(config=sample_config, directory_helper=directory_helper)

    mock_model.assert_called_once_with(
        sample_config.model.hf_model_ckpt,
        quantization_config=BitsAndBytesConfig(),
        use_cache=False,
        device_map=sample_config.model.device_map,
        torch_dtype=sample_config.model.casted_torch_dtype,
        attn_implementation=sample_config.model.attn_implementation,
    )

    mock_tokenizer.assert_called_once_with(sample_config.model.hf_model_ckpt)
    mock_inject_lora.assert_called_once()


def test_lora_injection(mocker):
    """Test the initialization of LoRAFinetune with a sample configuration."""
    # Mock dependencies that LoRAFinetune might call during initialization
    mocker.patch(
        "llmtune.finetune.lora.AutoModelForCausalLM.from_pretrained",
        return_value=MagicMock(),
    )
    mocker.patch(
        "llmtune.finetune.lora.AutoTokenizer.from_pretrained",
        return_value=MagicMock(),
    )

    mock_kbit = mocker.patch("llmtune.finetune.lora.prepare_model_for_kbit_training")
    mock_get_peft = mocker.patch("llmtune.finetune.lora.get_peft_model")

    # Initialize LoRAFinetune with the sample configuration
    LoRAFinetune(config=get_sample_config(), directory_helper=MagicMock())

    mock_kbit.assert_called_once()
    mock_get_peft.assert_called_once()


def test_model_finetune(mocker):
    sample_config = get_sample_config()

    mocker.patch(
        "llmtune.finetune.lora.AutoModelForCausalLM.from_pretrained",
        return_value=MagicMock(),
    )
    mocker.patch(
        "llmtune.finetune.lora.AutoTokenizer.from_pretrained", return_value=MagicMock()
    )
    mocker.patch(
        "llmtune.finetune.lora.LoRAFinetune._inject_lora",
        return_value=None,  # _inject_lora doesn't return a value
    )

    mock_trainer = mocker.MagicMock()
    mock_sft_trainer = mocker.patch(
        "llmtune.finetune.lora.SFTTrainer", return_value=mock_trainer
    )

    directory_helper = MagicMock()

    mock_training_args = mocker.patch(
        "llmtune.finetune.lora.TrainingArguments",
        return_value=MagicMock(),
    )

    ft = LoRAFinetune(config=sample_config, directory_helper=directory_helper)

    mock_dataset = MagicMock()
    ft.finetune(mock_dataset)

    mock_training_args.assert_called_once_with(
        logging_dir="/logs",
        output_dir=ft._weights_path,
        report_to="none",
        **sample_config.training.training_args.model_dump(),
    )

    mock_sft_trainer.assert_called_once_with(
        model=ft.model,
        train_dataset=mock_dataset,
        peft_config=ft._lora_config,
        tokenizer=ft.tokenizer,
        packing=True,
        args=mocker.ANY,  # You can replace this with the expected TrainingArguments if needed
        dataset_text_field="formatted_prompt",
        callbacks=mocker.ANY,  # You can replace this with the expected callbacks if needed
        **sample_config.training.sft_args.model_dump(),
    )

    mock_trainer.train.assert_called_once()


def test_save_model(mocker):
    # Prepare the configuration and directory helper
    sample_config = get_sample_config()

    # Mock dependencies that LoRAFinetune might call during initialization
    mocker.patch("llmtune.finetune.lora.AutoModelForCausalLM.from_pretrained")

    mock_tok = mocker.MagicMock()
    mocker.patch(
        "llmtune.finetune.lora.AutoTokenizer.from_pretrained", return_value=mock_tok
    )
    mocker.patch(
        "llmtune.finetune.lora.LoRAFinetune._inject_lora",
        return_value=None,
    )

    directory_helper = MagicMock()
    directory_helper.save_paths.weights = "/path/to/weights"

    mock_trainer = mocker.MagicMock()
    mocker.patch("llmtune.finetune.lora.SFTTrainer", return_value=mock_trainer)

    ft = LoRAFinetune(config=sample_config, directory_helper=directory_helper)

    mock_dataset = MagicMock()
    ft.finetune(mock_dataset)
    ft.save_model()

    mock_tok.save_pretrained.assert_called_once_with("/path/to/weights")
    mock_trainer.model.save_pretrained.assert_called_once_with("/path/to/weights")
