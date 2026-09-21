import sys, os, json, glob
sys.path.insert(0, r"C:\Users\yashm\OneDrive\Desktop\Deepfake-detection")

hub = os.path.expandvars(r"%USERPROFILE%\.cache\huggingface\hub")
configs = glob.glob(os.path.join(hub, r"models--*\**\config.json"), recursive=True)
for c in configs:
    cfg = json.load(open(c, encoding="utf-8"))
    i2l = cfg.get("id2label")
    l2i = cfg.get("label2id")
    print("MODEL:", os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(c)))))
    print("  id2label:", i2l)
    print("  label2id:", l2i)
