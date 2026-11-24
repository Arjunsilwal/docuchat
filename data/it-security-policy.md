# Acme Corp IT Security Policy

This policy protects Acme Corp's systems, data and customers. It applies to every employee and contractor with access to company systems. Violations may lead to disciplinary action.

## Passwords and Authentication

Passwords must be at least 14 characters long and must not be reused across systems. We recommend using a passphrase of four or more random words. All employees must store passwords in the company password manager, 1Password.

Multi-factor authentication (MFA) is mandatory for email, the VPN, source code repositories and every system that holds customer data. Use the authenticator app or a hardware security key; SMS codes are not allowed.

Passwords no longer need to be changed on a fixed schedule. You must change a password immediately if you think it may have been exposed.

### Password Reset

To reset a forgotten password, go to id.acme.example and choose "Forgot password". You will need your employee ID and a second factor. If you have lost your second factor as well, contact the IT Service Desk, which will verify your identity by video call before issuing a reset.

## VPN Access

The company VPN is required for all access to internal systems from outside the office, including from home. Install the VPN client from the Self Service portal. The VPN disconnects automatically after 12 hours, after which you sign in again with MFA.

Do not use public Wi-Fi without the VPN. Personal VPN services are not a replacement for the company VPN.

## Devices

Only company-managed laptops may access customer data. Personal phones may be used for email and chat only after installing the mobile device management (MDM) profile.

Laptops must have full-disk encryption enabled and must lock automatically after 5 minutes of inactivity. Never leave a device unattended in a public place.

### Lost or Stolen Devices

Report a lost or stolen device to the IT Service Desk within 1 hour of noticing. IT will remotely lock and wipe the device. Reporting quickly matters more than knowing exactly what happened.

## Data Classification

Acme Corp uses four data classes:

- **Public:** approved for anyone, such as marketing pages.
- **Internal:** for employees only, such as this policy.
- **Confidential:** limited to teams that need it, such as financial reports and contracts.
- **Restricted:** customer personal data and credentials. Restricted data must never be copied to personal devices, personal cloud storage or AI tools that are not on the approved list.

## Reporting Security Incidents

If you suspect a security incident, such as a phishing email you clicked, malware, or data sent to the wrong person, report it immediately. Open a ticket with the category SEC-7 in the IT Service Desk, or call the security hotline for urgent issues outside business hours.

Do not try to investigate the incident yourself and do not delete suspicious emails before the security team has reviewed them. Use the "Report phishing" button in your email client for suspected phishing.

## Security Training

All employees and contractors must complete security awareness training within their first 2 weeks and then once per year. Engineers must also complete secure coding training every year.