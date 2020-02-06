<?php
	header('Content-type: application/json');
	$status = array(
		'type'=>'success',
		'message'=>'Email sent!'
	);

    $name = $_POST['name']; 
    $email = @trim(stripslashes($_POST['email'])); 
    $message = @trim(stripslashes($_POST['message'])); 

    $email_from = $email;
    $email_to = 'hansonxufyp@gmail.com';

    $body = 'Name: ' . $name . "\n\n" . 'Email: ' . $email . "\n\n" . 'Message: ' . $message;

    $success = @mail($email_to,'Webpage Contact',$body . "\n\n" . 'From: <'.$email_from.'>');
    header("Location: https://hansonchang.com/idd.html");
	die;
?>