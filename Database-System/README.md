# Database System

Lab contents for the course "Database System".

Environment:

- Operating System: openEuler 24.03 LTS x86-64
- DBMS: openGauss 6.0.0 build compiled at 2025-12-25 20:12:07 commit 0 last mr  release (Lite)

## Setup Guide

### Run openEuler OS in a Virtual Machine

Recommend to disable the firewall service. If not, you might have to manually open some ports in the following procedures.

Disable the firewall service:

```bash
sudo systemctl stop firewalld.service
sudo systemctl disable firewalld.service
```

### Install openGauss and Enter the Database

openGauss installation:

```bash
sudo yum install opengauss -y
```

Start the database server:

```bash
gs_ctl start -D $PGDATA
```

> - `$PGDATA`: Default environment variable path. In default it would expand to `/var/lib/opengauss/data`.

Connect to the database:

```bash
gsql -d postgres -p 7654 -r
```

or

```bash
gsql -d postgres
```

The default port number is 7654. After the database is connected, the database can be properly used.

### Set the Password

Once entered the database system ,use the command:

```SQL
ALTER ROLE opengauss IDENTIFIED BY 'YourPassword@123';
```

### Configure the Remote Connection

You may need to change the user password of user `opengauss` and add it to the sudoers before the following commands.

You would have to create a new database system user to allow remote connection. openGauss would refuse the remote connection to the initial users.

#### Create a New User

Enter the database and create a new user with password:

```bash
gsql -d postgres
CREATE USER dbuser WITH PASSWORD 'YourRemoteDb@2026';
ALTER USER dbuser SYSADMIN; # Set as admin
```

#### Edit `postgresql.conf`

Enable listening to IP addresses other than localhost.

Locate to `$PGDATA/postgresql.conf`, find `listen_addresses`, and modify it to:

```text
listen_addresses = '*'
```

Find `password_encryption_type`, make sure it is set to `1`. `1` is allowing both `md5` and `sha256`.

Make sure both lines are uncommented.

#### Edit `pg_hba.conf`

Configure the whitelist to allow IP addresses of the remote clients.

Locate to `$PGDATA/pg_hba.conf`, add the following rule at the bottom of the file:

```text
host    all             dbuser             0.0.0.0/0               sha256
```

*(allowing all the IP adresses to connect), replace the `dbuser` with the actual user name you set before.*

> **Important:**
>
> Some of the clients may use the md5 encryption type. In this case, change the `sha256` above to `md5`. And AFTER THAT, you MUST change the password of the user in the database server again to regenerate the encrypted password using the new encryption type.

#### Optional: Open Firewall Ports

Run the command using `root` user:

```bash
sudo firewall-cmd --zone=public --add-port=26000/tcp --permanent
sudo firewall-cmd --reload
```

#### Restart openGauss

```bash
gs_ctl restart -D $PGDATA
```

### Connect Remotely to the Database

Use a client software (e.g. navicat) to connect to the client. Use this command to see current network configuration of opengauss:

```bash
netstat -lnpt | grep gauss
```

## Some Useful Commands

- `su - opengauss`: Switch to the user opengauss.

> The following commands needs to be run using the `opengauss` user.

- `ps ux`: Check the process. The binary installation directory is /usr/local/opengauss, and the default startup data directory is /var/lib/opengauss/data.
- `gs_ctl status -D $PGDATA`: Check status of the database.
- `gs_ctl start -D $PGDATA`: Start the database server.
- `gs_ctl restart -D $PGDATA`: Restart the database server.
- `gsql -d postgres -p 7654 -r`: Connect to the database.


> The following commands needs to be run after entering the database.

- `\q`: Exit the database.
- `\du`: Display all the users.
- `ALTER ROLE opengauss IDENTIFIED BY 'YourPassword@123';`: Change the administrator password.
- `CREATE USER dbuser WITH PASSWORD 'YourRemoteDb@2026';`: Add a new user with password.
- `ALTER USER dbuser SYSADMIN;`: Set a user as admin.
