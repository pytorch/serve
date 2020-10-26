package org.pytorch.serve.plugins.ddb;

import software.amazon.awssdk.enhanced.dynamodb.DynamoDbEnhancedClient;

public final class DDBClient {
    private static DynamoDbEnhancedClient client;

    private DDBClient() {}

    public static synchronized DynamoDbEnhancedClient getInstance() {
        if (client == null) {
            client = DynamoDbEnhancedClient.create();
        }
        return client;
    }
}
