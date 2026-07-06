import unittest
from pathlib import Path

from unimeth.config import model_config
from unimeth.config.model_config import ModelConfig


class ModelConfigPackageDataTest(unittest.TestCase):
    def test_named_configs_are_loaded_from_package_data(self):
        package_root = Path(model_config.__file__).resolve().parents[1]
        expected_config_dir = package_root / "configs"

        self.assertEqual(model_config.get_config_dir(), expected_config_dir)
        self.assertTrue((expected_config_dir / "default.json").exists())
        self.assertTrue((expected_config_dir / "distilled.json").exists())

        self.assertEqual(ModelConfig.from_name("default").d_model, 384)
        self.assertEqual(ModelConfig.from_name("distilled").d_model, 256)


if __name__ == "__main__":
    unittest.main()
