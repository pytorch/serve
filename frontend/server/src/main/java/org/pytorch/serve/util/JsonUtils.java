package org.pytorch.serve.util;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

public final class JsonUtils {

    public static final Gson GSON_PRETTY =
            new GsonBuilder()
                    .setDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")
                    .setPrettyPrinting()
                    .create();
    public static final Gson GSON = new GsonBuilder().create();

    private JsonUtils() {}
}
