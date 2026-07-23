import subprocess
def test_config_has_clip_model():
    out = subprocess.run(
        ["node", "-e", "import('./src/semdb/semdb.config.mjs').then(m=>console.log(!!m.defaults.extraction.clipModel))"],
        cwd="/local-scratch/localhome/hza214/GenDB", capture_output=True, text=True)
    assert out.stdout.strip() == "true", out.stderr
