import json
from pathlib import Path
import unittest

from ecthr_benchmark.adaptation import validate_adaptation, validate_module_names


class _NamedModel:
    def named_modules(self):
        return iter([
            ("", object()),
            ("transformer.h.0.attn.attention.q_proj", object()),
            ("score", object()),
        ])


class AdaptationTests(unittest.TestCase):
    def test_checked_configs_are_valid_and_comparable(self):
        lora = json.loads(Path("configs/gpt_neo_125m_lora.json").read_text())
        qlora = json.loads(Path("configs/gpt_neo_125m_qlora.json").read_text())
        lora_staging = json.loads(Path("configs/gpt_neo_125m_lora_staging.json").read_text())
        qlora_staging = json.loads(Path("configs/gpt_neo_125m_qlora_staging.json").read_text())
        validate_adaptation(lora)
        validate_adaptation(qlora)
        validate_adaptation(lora_staging)
        validate_adaptation(qlora_staging)
        self.assertEqual(lora["lora"], qlora["lora"])
        self.assertEqual(lora["model_revision"], qlora["model_revision"])
        self.assertEqual(lora_staging["lora"], qlora_staging["lora"])
        self.assertEqual(lora_staging["max_steps"], qlora_staging["max_steps"])

    def test_adaptation_requires_pinned_revision_and_explicit_targets(self):
        base = json.loads(Path("configs/gpt_neo_125m_lora.json").read_text())
        with self.assertRaisesRegex(ValueError, "immutable"):
            validate_adaptation({**base, "model_revision": "main"})
        broken = {**base, "lora": {**base["lora"], "target_modules": []}}
        with self.assertRaisesRegex(ValueError, "target_modules"):
            validate_adaptation(broken)

    def test_qlora_requires_lora_cuda_path_and_supported_4bit_values(self):
        base = json.loads(Path("configs/gpt_neo_125m_qlora.json").read_text())
        with self.assertRaisesRegex(ValueError, "lora object"):
            validate_adaptation({**base, "lora": None})
        with self.assertRaisesRegex(ValueError, "CUDA"):
            validate_adaptation({**base, "use_cpu": True})
        broken = {**base, "quantization": {**base["quantization"], "quant_type": "int3"}}
        with self.assertRaisesRegex(ValueError, "nf4 or fp4"):
            validate_adaptation(broken)

    def test_module_names_are_checked_but_all_linear_is_supported(self):
        validate_module_names(_NamedModel(), {"target_modules": "all-linear", "modules_to_save": ["score"]})
        validate_module_names(_NamedModel(), {"target_modules": ["q_proj"], "modules_to_save": ["score"]})
        with self.assertRaisesRegex(ValueError, "target module"):
            validate_module_names(_NamedModel(), {"target_modules": ["missing"]})
        with self.assertRaisesRegex(ValueError, "modules_to_save"):
            validate_module_names(_NamedModel(), {"target_modules": "all-linear", "modules_to_save": ["missing"]})


if __name__ == "__main__":
    unittest.main()
