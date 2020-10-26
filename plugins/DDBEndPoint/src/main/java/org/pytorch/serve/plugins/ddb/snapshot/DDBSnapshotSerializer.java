package org.pytorch.serve.plugins.ddb.snapshot;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.IOException;
import java.io.StringReader;
import java.io.StringWriter;
import java.text.SimpleDateFormat;
import java.time.LocalDate;
import java.util.Date;
import java.util.Iterator;
import java.util.Properties;
import org.pytorch.serve.plugins.ddb.DDBClient;
import org.pytorch.serve.servingsdk.snapshot.Snapshot;
import org.pytorch.serve.servingsdk.snapshot.SnapshotSerializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import software.amazon.awssdk.enhanced.dynamodb.DynamoDbIndex;
import software.amazon.awssdk.enhanced.dynamodb.DynamoDbTable;
import software.amazon.awssdk.enhanced.dynamodb.TableSchema;
import software.amazon.awssdk.enhanced.dynamodb.mapper.StaticAttributeTags;
import software.amazon.awssdk.enhanced.dynamodb.model.Page;
import software.amazon.awssdk.enhanced.dynamodb.model.QueryConditional;
import software.amazon.awssdk.enhanced.dynamodb.model.QueryEnhancedRequest;
import software.amazon.awssdk.services.dynamodb.model.DynamoDbException;
import software.amazon.awssdk.services.dynamodb.model.ResourceNotFoundException;

public class DDBSnapshotSerializer implements SnapshotSerializer {
    private Logger logger = LoggerFactory.getLogger(DDBSnapshotSerializer.class);

    private static final String MODEL_SNAPSHOT = "model_snapshot";
    private static final String SNAPSHOT_NAME = "snapshotName";
    private static final String SNAPSHOT_CREATED_ON = "createdOn";
    private static final String SNAPSHOT_CREATED_ON_MONTH = "createdOnMonth";
    private static final String SNAPSHOT_DATA = "snapshot";
    private static final String TABLE_NAME = "Snapshots";

    public static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
    private static TableSchema<Snapshots> snapshotTableSchema;

    public DDBSnapshotSerializer() {
        snapshotTableSchema = getTableSchema();
    }

    @Override
    public void saveSnapshot(Snapshot snapshot, final Properties prop) throws IOException {

        logger.info("Saving snapshot to DDB...");
        try {
            DynamoDbTable<Snapshots> ddbSnapshotTable =
                    DDBClient.getInstance().table(TABLE_NAME, snapshotTableSchema);

            Date createdOnDt = new Date(snapshot.getCreated());
            SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss,SSS");
            String createdOn = sdf.format(createdOnDt);

            SimpleDateFormat monthFmt = new SimpleDateFormat("yyyy-MM");
            String createdOnMonth = monthFmt.format(createdOnDt);

            String snapshotJson = GSON.toJson(snapshot, Snapshot.class);
            prop.put(MODEL_SNAPSHOT, snapshotJson);

            Snapshots ddbSnapshot = new Snapshots();
            ddbSnapshot.setSnapshotName(getKeyVal(snapshot));
            ddbSnapshot.setSnapshot(convert2String(prop));
            ddbSnapshot.setCreatedOn(createdOn);
            ddbSnapshot.setCreatedOnMonth(createdOnMonth);

            ddbSnapshotTable.putItem(ddbSnapshot);

        } catch (ResourceNotFoundException e) {
            logger.error("Snapshot serialization error: The table {} can't be found", TABLE_NAME);
            logger.error("Be sure that DDB table exists and that you've typed its name correctly");
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
        Properties snapshot = null;
        try {
            DynamoDbTable<Snapshots> snapshotsTable =
                    DDBClient.getInstance().table(TABLE_NAME, snapshotTableSchema);

            String currMonthYr =
                    String.format(
                            "%s-%s", LocalDate.now().getYear(), LocalDate.now().getMonthValue());
            QueryConditional queryConditional =
                    QueryConditional.keyEqualTo(k -> k.partitionValue(currMonthYr));
            DynamoDbIndex<Snapshots> createdOnMonth = snapshotsTable.index("createdOnMonth-index");
            QueryEnhancedRequest query =
                    QueryEnhancedRequest.builder()
                            .consistentRead(false)
                            .limit(1)
                            .queryConditional(queryConditional)
                            .scanIndexForward(false)
                            .build();

            Iterator<Page<Snapshots>> results = createdOnMonth.query(query).iterator();
            if (!results.next().items().isEmpty()) {
                Snapshots lastSnapshot = results.next().items().get(0);
                snapshot = convert2Properties(lastSnapshot.getSnapshot());
                logger.info(
                        "The last snapshot name obtained from DDB is {}",
                        lastSnapshot.getSnapshotName());
            } else {
                logger.error(
                        "Failed to get last snpahost from DDB. Torchserve will start with default or given configuration.");
            }
        } catch (ResourceNotFoundException e) {
            logger.error(
                    "DDB error while getting last snapshot. The table {} can't be found. . Shutting down.",
                    TABLE_NAME);
            logger.error("Be sure that DDB table exists and that you've typed its name correctly");
            System.exit(1);
        } catch (DynamoDbException e) {
            logger.error(
                    "DDB error while getting last snapshot. Shutting down. Details: {}",
                    e.getMessage());
            System.exit(1);
        } catch (IOException e) {
            logger.error("Snapshot de-serialization error: {}", e.getMessage());
            System.exit(1);
        }

        return snapshot;
    }

    private static TableSchema<Snapshots> getTableSchema() {
        TableSchema<Snapshots> schema =
                TableSchema.builder(Snapshots.class)
                        .newItemSupplier(Snapshots::new)
                        .addAttribute(
                                String.class,
                                a ->
                                        a.name(SNAPSHOT_NAME)
                                                .getter(Snapshots::getSnapshotName)
                                                .setter(Snapshots::setSnapshotName)
                                                .tags(StaticAttributeTags.primaryPartitionKey()))
                        .addAttribute(
                                String.class,
                                a ->
                                        a.name(SNAPSHOT_CREATED_ON)
                                                .getter(Snapshots::getCreatedOn)
                                                .setter(Snapshots::setCreatedOn)
                                                .tags(StaticAttributeTags.primarySortKey())
                                                .tags(
                                                        StaticAttributeTags.secondarySortKey(
                                                                "createdOnMonth-index")))
                        .addAttribute(
                                String.class,
                                a ->
                                        a.name(SNAPSHOT_CREATED_ON_MONTH)
                                                .getter(Snapshots::getCreatedOnMonth)
                                                .setter(Snapshots::setCreatedOnMonth)
                                                .tags(
                                                        StaticAttributeTags.secondaryPartitionKey(
                                                                "createdOnMonth-index")))
                        .addAttribute(
                                String.class,
                                a ->
                                        a.name(SNAPSHOT_DATA)
                                                .getter(Snapshots::getSnapshot)
                                                .setter(Snapshots::setSnapshot))
                        .build();
        return schema;
    }

    private static String getKeyVal(final Snapshot snapshot) {
        return snapshot.getName() + "_" + snapshot.getCreated() + "_" + snapshot.getModelCount();
    }

    private static String convert2String(final Properties props) throws IOException {
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
