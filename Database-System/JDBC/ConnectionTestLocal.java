import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;

public class ConnectionTestLocal {
    private static final String DEFAULT_URL = "jdbc:opengauss://192.168.39.160:7654/mydb";
    private static final String DEFAULT_USER = "dbremote";
    private static final String DEFAULT_PASSWORD = "dbremote:399";

    public static void main(String[] args) {
        String url = args.length > 0 ? args[0] : DEFAULT_URL;
        String user = args.length > 1 ? args[1] : DEFAULT_USER;
        String password = args.length > 2 ? args[2] : DEFAULT_PASSWORD;

        try {
            Class.forName("org.opengauss.Driver");
        } catch (ClassNotFoundException e) {
            System.err.println("Failed to load openGauss JDBC driver: " + e.getMessage());
            return;
        }

        String sql = "SELECT current_database(), current_user, version()";

        try (Connection connection = DriverManager.getConnection(url, user, password);
             PreparedStatement statement = connection.prepareStatement(sql);
             ResultSet resultSet = statement.executeQuery()) {

            if (resultSet.next()) {
                System.out.println("Connection successful.");
                System.out.println("Database: " + resultSet.getString(1));
                System.out.println("User: " + resultSet.getString(2));
                System.out.println("Server version: " + resultSet.getString(3));
            } else {
                System.out.println("Connection successful, but the verification query returned no rows.");
            }
        } catch (SQLException e) {
            System.err.println("Connection test failed.");
            System.err.println("SQLState: " + e.getSQLState());
            System.err.println("ErrorCode: " + e.getErrorCode());
            System.err.println(e.getMessage());
        }
    }
}