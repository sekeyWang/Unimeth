import unittest
from pathlib import Path
from unittest.mock import patch

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

    def test_config_dir_rejects_non_filesystem_resources(self):
        class FakeTraversable:
            pass

        with patch.object(model_config.resources, "files", return_value=FakeTraversable()):
            with self.assertRaises(RuntimeError) as ctx:
                model_config.get_config_dir()

        self.assertIn("not available as a filesystem path", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
