package org.pytorch.serve.metrics.configuration;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import org.yaml.snakeyaml.Yaml;
import org.yaml.snakeyaml.composer.ComposerException;
import org.yaml.snakeyaml.constructor.Constructor;

public final class YAMLMetricConfigurationHandler {
    private YAMLMetricConfigurationHandler() {
        throw new UnsupportedOperationException();
    }

    static MetricConfiguration loadConfiguration(String configFilePath)
            throws FileNotFoundException, ComposerException, RuntimeException {
        Constructor constructor = new Constructor(MetricConfiguration.class);
        Yaml yaml = new Yaml(constructor);
        FileInputStream inputStream = new FileInputStream(new File(configFilePath));
        MetricConfiguration config = yaml.load(inputStream);
        config.validate();

        return config;
    }
}
