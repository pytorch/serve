package org.pytorch.serve.plugins.snapshot;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.IOException;
import java.io.StringReader;
import java.io.StringWriter;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashMap;
import java.util.Properties;
import java.util.Set;
import org.pytorch.serve.servingsdk.snapshot.Snapshot;
import org.pytorch.serve.servingsdk.snapshot.SnapshotSerializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import software.amazon.awssdk.services.dynamodb.DynamoDbClient;
import software.amazon.awssdk.services.dynamodb.model.AttributeAction;
import software.amazon.awssdk.services.dynamodb.model.AttributeValue;
import software.amazon.awssdk.services.dynamodb.model.AttributeValueUpdate;
import software.amazon.awssdk.services.dynamodb.model.DynamoDbException;
import software.amazon.awssdk.services.dynamodb.model.GetItemRequest;
import software.amazon.awssdk.services.dynamodb.model.GetItemResponse;
import software.amazon.awssdk.services.dynamodb.model.PutItemRequest;
import software.amazon.awssdk.services.dynamodb.model.ResourceNotFoundException;
import software.amazon.awssdk.services.dynamodb.model.UpdateItemRequest;

public class DDBSnapshotSerializer implements SnapshotSerializer {
    private Logger logger = LoggerFactory.getLogger(DDBSnapshotSerializer.class);
    private static final String MODEL_SNAPSHOT = "model_snapshot";
    public static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

    private static final String TABLE_NAME = "snapshots";
    private static final String TABLE_NAME_RECENTLY_ADDED = "latest_snapshots";
    private static final String LATEST_SNAPSHOT_KEY = "latestSnapshot";
    private static final String PKEY = "snapshotName";
    private static final String SNAPSHOT = "snapshot";
    private static final String CREATED_ON = "createdOn";

    private DynamoDbClient client;

    public DDBSnapshotSerializer() {
        client = DDBSnapshotSerializer.createClient();
    }

    private static DynamoDbClient createClient() {
        return DynamoDbClient.builder().build();
    }

    @Override
    public void saveSnapshot(Snapshot snapshot, final Properties prop) throws IOException {

        logger.info("Saving snapshot to DDB...");

        String snapshotJson = GSON.toJson(snapshot, Snapshot.class);
        prop.put(MODEL_SNAPSHOT, snapshotJson);

        Date d = new Date(snapshot.getCreated());
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss,SSS");
        String createdOn = sdf.format(d);

        putItemInTable(getKeyVal(snapshot), convert2String(prop), createdOn);
    }

    private void putItemInTable(String pKeyVal, String snapshot, String createdOn) {

        HashMap<String, AttributeValue> itemValues = new HashMap<String, AttributeValue>();

        itemValues.put(PKEY, AttributeValue.builder().s(pKeyVal).build());
        itemValues.put(SNAPSHOT, AttributeValue.builder().s(snapshot).build());
        itemValues.put(CREATED_ON, AttributeValue.builder().s(createdOn).build());

        PutItemRequest putItemrequest =
                PutItemRequest.builder().tableName(TABLE_NAME).item(itemValues).build();

        HashMap<String, AttributeValue> itemKey = new HashMap<String, AttributeValue>();
        itemKey.put(PKEY, AttributeValue.builder().s(LATEST_SNAPSHOT_KEY).build());

        HashMap<String, AttributeValueUpdate> updatedValues =
                new HashMap<String, AttributeValueUpdate>();
        updatedValues.put(
                SNAPSHOT,
                AttributeValueUpdate.builder()
                        .value(AttributeValue.builder().s(snapshot).build())
                        .action(AttributeAction.PUT)
                        .build());

        UpdateItemRequest updateRequest =
                UpdateItemRequest.builder()
                        .tableName(TABLE_NAME_RECENTLY_ADDED)
                        .key(itemKey)
                        .attributeUpdates(updatedValues)
                        .build();

        try {
            client.updateItem(updateRequest);
            client.putItem(putItemrequest);
            logger.info("DDB was successfully updated");

        } catch (ResourceNotFoundException e) {
            logger.error("Snapshot serialization error: The table {} can't be found", TABLE_NAME);
            logger.error("Be sure that it exists and that you've typed its name correctly");
        } catch (DynamoDbException e) {
            logger.error("Snapshot serialization error: {}", e.getMessage());
        }
    }

    @Override
    public Snapshot getSnapshot(String snapshotJson) throws IOException {
        return GSON.fromJson(snapshotJson, Snapshot.class);
    }

    @Override
    public Properties getLastSnapshot() {
        logger.info("Fetching last snapshot from DDB...");
        HashMap<String, AttributeValue> getItemValues = new HashMap<String, AttributeValue>();
        getItemValues.put(PKEY, AttributeValue.builder().s(LATEST_SNAPSHOT_KEY).build());

        GetItemRequest request =
                GetItemRequest.builder()
                        .tableName(TABLE_NAME_RECENTLY_ADDED)
                        .key(getItemValues)
                        .build();

        try {
            GetItemResponse response = client.getItem(request);
            if (response.hasItem()) {
                Set<String> keys = response.item().keySet();

                for (String attribute : keys) {
                    String value = response.item().get(attribute).toString();
                    if (SNAPSHOT.equalsIgnoreCase(attribute)) {

                        Properties prop = convert2Properties(value);
                        logger.info("Successfully fetched last snapshot from DDB");
                        return prop;
                    }
                }
            } else {
                logger.error("No item found with the key {}", PKEY);
            }

        } catch (DynamoDbException e) {
            logger.error("Snapshot de-serialization error: {}", e.getMessage());
        } catch (IOException e) {
            logger.error("Failed to build properties from DDB snapshot");
        }
        return null;
    }

    private String getKeyVal(final Snapshot snapshot) {
        return snapshot.getName() + "_" + snapshot.getCreated() + "_" + snapshot.getModelCount();
    }

    private String convert2String(final Properties props) throws IOException {
        final StringWriter sw = new StringWriter();
        String propStr;
        try {
            props.store(sw, "snapshot");
            propStr = sw.toString();
        } finally {
            if (sw != null) {
                sw.close();
            }
        }

        return propStr;
    }

    private static Properties convert2Properties(final String propsStr) throws IOException {
        final Properties props = new Properties();
        final StringReader sr = new StringReader(propsStr);
        try {
            props.load(sr);
        } finally {
            if (sr != null) {
                sr.close();
            }
        }

        return props;
    }
}
