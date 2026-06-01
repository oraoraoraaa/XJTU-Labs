import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;

public class TriggerDeletionTask {
    private static final String URL = "jdbc:opengauss://192.168.39.160:7654/mydb";
    private static final String USER = "dbremote";
    private static final String PASSWORD = "dbremote:399";

    // Helper class to temporarily hold composite primary keys
    static class SCRef {
        String sNum;
        String cNum;
        SCRef(String sNum, String cNum) {
            this.sNum = sNum;
            this.cNum = cNum;
        }
    }

    public static void main(String[] args) {
        try {
            Class.forName("org.opengauss.Driver");
        } catch (ClassNotFoundException e) {
            System.err.println("Driver loading failed: " + e.getMessage());
            return;
        }

        String selectSql = "SELECT \"S_num\", \"C_num\" FROM \"public\".\"SC799\" WHERE \"GRADE\" IS NULL ORDER BY random() LIMIT 100";
        String deleteSql = "DELETE FROM \"public\".\"SC799\" WHERE \"S_num\" = ? AND \"C_num\" = ?";

        List<SCRef> targets = new ArrayList<>();

        try (Connection conn = DriverManager.getConnection(URL, USER, PASSWORD)) {
            System.out.println("Connected to mydb successfully.");

            // 1. Fetch 100 random records with NULL grades
            try (PreparedStatement selectStmt = conn.prepareStatement(selectSql);
                 ResultSet rs = selectStmt.executeQuery()) {
                while (rs.next()) {
                    targets.add(new SCRef(rs.getString("S_num"), rs.getString("C_num")));
                }
            }

            System.out.println("Found " + targets.size() + " records to delete.");

            // 2. Perform sequential, one-by-one deletions
            int deletedCount = 0;
            try (PreparedStatement deleteStmt = conn.prepareStatement(deleteSql)) {
                for (SCRef ref : targets) {
                    deleteStmt.setString(1, ref.sNum);
                    deleteStmt.setString(2, ref.cNum);
                    
                    int rowsAffected = deleteStmt.executeUpdate();
                    if (rowsAffected > 0) {
                        deletedCount++;
                    }
                }
            }

            System.out.println("Successfully deleted " + deletedCount + " records one by one.");
            System.out.println("The trigger has automatically backed them up into the BK799 table.");

        } catch (SQLException e) {
            System.err.println("Database operation failed.");
            e.printStackTrace();
        }
    }
}