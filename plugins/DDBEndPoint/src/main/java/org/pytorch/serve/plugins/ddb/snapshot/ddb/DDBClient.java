package org.pytorch.serve.plugins.ddb.snapshot.ddb;

import software.amazon.awssdk.enhanced.dynamodb.DynamoDbEnhancedClient;

public final class DDBClient {
    private static DynamoDbEnhancedClient client;

    private DDBClient() {}

    public static synchronized DynamoDbEnhancedClient initInstance() {
        if (DDBClient.client == null) {
            DDBClient.client = DynamoDbEnhancedClient.create();
        }
        return DDBClient.client;
    }

    public static synchronized void initInstance(DynamoDbEnhancedClient client) {
        if (DDBClient.client == null) {
            DDBClient.client = client;
        }
    }
}
