package org.pytorch.serve.servingsdk.impl;

import java.util.HashMap;
import java.util.Map;
import java.util.Properties;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.wlm.ModelManager;
import software.amazon.ai.mms.servingsdk.Context;
import software.amazon.ai.mms.servingsdk.Model;

public class ModelServerContext implements Context {
    @Override
    public Properties getConfig() {
        return ConfigManager.getInstance().getConfiguration();
    }

    @Override
    public Map<String, Model> getModels() {
        HashMap<String, Model> r = new HashMap<>();
        ModelManager.getInstance().getModels().forEach((k, v) -> r.put(k, new ModelServerModel(v)));
        return r;
    }
}
